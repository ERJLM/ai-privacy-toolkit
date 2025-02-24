import numpy as np
import pandas as pd
from collections import Counter

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from apt.utils.datasets import ArrayDataset, DATA_PANDAS_NUMPY_TYPE
from scipy.stats import wasserstein_distance
from typing import Union, Optional


class Anonymize:
    """
    Class for performing tailored, model-guided anonymization of training datasets for ML models.

    Based on the implementation described in: https://arxiv.org/abs/2007.13086

    :param k: The privacy parameter that determines the number of records that will be indistinguishable from each
              other (when looking at the quasi identifiers). Should be at least 2.
    :type k: int
    :param quasi_identifiers: The features that need to be minimized in case of pandas data, and indexes of features
                              in case of numpy data.
    :type quasi_identifiers: np.ndarray or list of strings or integers.
    :param quasi_identifer_slices: If some of the quasi-identifiers represent 1-hot encoded features that need to remain
                                   consistent after anonymization, provide a list containing the list of column names
                                   or indexes that represent a single feature.
    :type quasi_identifer_slices: list of lists of strings or integers.
    :param categorical_features: The list of categorical features (if supplied, these featurtes will be one-hot encoded
                                 before using them to train the decision tree model).
    :type categorical_features: list, optional
    :param is_regression: Whether the model is a regression model or not (if False, assumes a classification model).
                          Default is False.
    :type is_regression: list, optional
    :param train_only_QI: The required method to train data set for anonymization. Default is
                          to train the tree on all features.
    :type train_only_QI: boolean, optional
    :type l: int, optional
    :param l: The l-diversity parameter that determines the minimum number of distinct sensitive values in each group. Should be at least 1.
    :type t: float, optional
    :param t: The t-closeness parameter that ensures that the sensitive data within each group is statistically similar to the sensitive data within the whole dataset.
              Should be between 0 and 1.
    :type sensitive_attributes, optional
    :param sensitive_attributes: List containing all the sensitive attributes.
    """

    def __init__(self, k: int,
                 quasi_identifiers: Union[np.ndarray, list],
                 quasi_identifer_slices: Optional[list] = None,
                 categorical_features: Optional[list] = None,
                 is_regression: Optional[bool] = False,
                 train_only_QI: Optional[bool] = False,
                 l: Optional[int] = None,  
                 t: Optional[float] = None,  
                 sensitive_attributes: Optional[list] = None): 
        if k < 2:
            raise ValueError("k should be a positive integer with a value of 2 or higher")
        if l and l < 1:
            raise ValueError("l should be at least 1")
        if t and not (0 <= t <= 1):
            raise ValueError("t should be between 0 and 1")
        if quasi_identifiers is None or len(quasi_identifiers) < 1:
            raise ValueError("The list of quasi-identifiers cannot be empty")
        if (l or t) and not sensitive_attributes:
            raise ValueError("Sensitive attribute must be provided for l-diversity and t-closeness")

        self.k = k
        self.l = l
        self.t = t
        self.sensitive_attributes = sensitive_attributes
        self.quasi_identifiers = quasi_identifiers
        self.categorical_features = categorical_features
        self.is_regression = is_regression
        self.train_only_QI = train_only_QI
        self.features_names = None
        self.features = None
        self.quasi_identifer_slices = quasi_identifer_slices
        self.global_distribution = None

    def anonymize(self, dataset: ArrayDataset) -> DATA_PANDAS_NUMPY_TYPE:
        """
        Method for performing model-guided anonymization.

        :param dataset: Data wrapper containing the training data for the model and the predictions of the
                        original model on the training data.
        :type dataset: `ArrayDataset`
        :return: The anonymized training dataset as either numpy array or pandas DataFrame (depending on the type of
                 the original data used to create the ArrayDataset).
        """
        if dataset.get_samples().shape[1] != 0:
            self.features = [i for i in range(dataset.get_samples().shape[1])]
        else:
            raise ValueError('No data provided')

        if dataset.features_names is not None:
            self.features_names = dataset.features_names
        else:  # if no names provided, use numbers instead
            self.features_names = self.features

        if not set(self.quasi_identifiers).issubset(set(self.features_names)):
            raise ValueError('Quasi identifiers should bs a subset of the supplied features or indexes in range of '
                             'the data columns')
        if self.categorical_features and not set(self.categorical_features).issubset(set(self.features_names)):
            raise ValueError('Categorical features should bs a subset of the supplied features or indexes in range of '
                             'the data columns')
        
        # Get the indices of the sensitive attributes    
        if self.sensitive_attributes:    
            self.sensitive_indices = [self.features_names.index(sensitive_attribute) for sensitive_attribute in self.sensitive_attributes]
        
        # Calculate global distribution for t-closeness
        if self.t is not None:
            self._calculate_global_distribution(dataset.get_samples())
            
        # transform quasi identifiers to indexes
        self.quasi_identifiers = [i for i, v in enumerate(self.features_names) if v in self.quasi_identifiers]
        if self.quasi_identifer_slices:
            temp_list = []
            for slice in self.quasi_identifer_slices:
                new_slice = [i for i, v in enumerate(self.features_names) if v in slice]
                temp_list.append(new_slice)
            self.quasi_identifer_slices = temp_list
        if self.categorical_features:
            self.categorical_features = [i for i, v in enumerate(self.features_names) if v in self.categorical_features]

        transformed = self._anonymize(dataset.get_samples().copy(), dataset.get_labels())
        if dataset.is_pandas:
            return pd.DataFrame(transformed, columns=self.features_names)
        else:
            return transformed

    def _anonymize(self, x, y):
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y should have same number of rows")
        if x.dtype.kind not in 'iufc':
            if not self.categorical_features:
                raise ValueError('when supplying an array with non-numeric data, categorical_features must be defined')
            x_prepared = self._modify_categorical_features(x)
        else:
            x_prepared = x
        x_anonymizer_train = x_prepared
        if self.train_only_QI:
            # build DT just on QI features
            x_anonymizer_train = x_prepared[:, self.quasi_identifiers]
        if self.is_regression:
            self._anonymizer = DecisionTreeRegressor(random_state=10, min_samples_split=2, min_samples_leaf=self.k)
        else:
            self._anonymizer = DecisionTreeClassifier(random_state=10, min_samples_split=2, min_samples_leaf=self.k)

        self._anonymizer.fit(x_anonymizer_train, y)
        cells_by_id = self._calculate_cells(x, x_anonymizer_train)
        return self._anonymize_data(x, x_anonymizer_train, cells_by_id)

    def _calculate_cells(self, x, x_anonymizer_train):
        # x is original data, x_anonymizer_train is only QIs + 1-hot encoded
        cells_by_id = {}
        leaves = []
        for node, feature in enumerate(self._anonymizer.tree_.feature):
            if feature == -2:  # leaf node
                leaves.append(node)
                hist = [int(i) for i in self._anonymizer.tree_.value[node][0]]
                # TODO we may change the method for choosing representative for cell
                # label_hist = self.anonymizer.tree_.value[node][0]
                # label = int(self.anonymizer.classes_[np.argmax(label_hist)])
                cell = {'label': 1, 'hist': hist, 'id': int(node)}
                cells_by_id[cell['id']] = cell
        self._nodes = leaves
        self._find_representatives(x, x_anonymizer_train, cells_by_id.values())
        return cells_by_id
    
    def _check_l_diversity(self, rows: np.ndarray) -> bool:
        """Checks if a group satisfies l-diversity for all sensitive attributes."""
        
        if self.l is None:
            return True
        
        # Create a variable to store the number of unique sensitive values
        n_unique = 0
        
        # Get the number of unique sensitive values
        for sensitive_index in self.sensitive_indices:
            n_unique += len(set(rows[:, sensitive_index]))
        return n_unique >= self.l
        
        return True
    
    def _calculate_global_distribution(self, data: np.ndarray):
        """Calculate the global distribution of sensitive attributes."""
        self.global_distribution = {}
        for idx in self.sensitive_indices:
            values, counts = np.unique(data[:, idx], return_counts=True)
            self.global_distribution[idx] = dict(zip(values, counts / len(data)))

    def _check_t_closeness(self, rows: np.ndarray) -> bool:
        """Checks if a group satisfies t-closeness for all sensitive attributes."""
    
        if self.t is None:
            return True

        for idx in self.sensitive_indices:
        
            # Get the group distribution
            group_values, group_counts = np.unique(rows[:, idx], return_counts=True)
            group_dist = group_counts / len(rows)
        
            # Get the global distribution
            global_values = np.array(list(self.global_distribution[idx].keys()))
            global_dist = np.array(list(self.global_distribution[idx].values()))
            
            # If the values are not numerical, map the unique indices to the values
            if not np.issubdtype(group_values.dtype, np.number):
                all_values = np.concatenate([group_values.astype(str), global_values.astype(str)])
                category_mapping = {val: idx for idx, val in enumerate(np.unique(all_values))} 
                group_values = np.array([category_mapping.get(val, -1) for val in group_values])
                global_values = np.array([category_mapping.get(val, -1) for val in global_values])
            
            # Calculate the EMD
            emd = wasserstein_distance(group_values, global_values, u_weights=group_dist, v_weights=global_dist)
            
            if emd > self.t:
                return False
        
        return True

    def _find_representatives(self, x, x_anonymizer_train, cells):
        
        # x is original data (always numpy), x_anonymizer_train is only QIs + 1-hot encoded
        node_ids = self._find_sample_nodes(x_anonymizer_train)
        if self.quasi_identifer_slices:
            all_one_hot_features = set([feature for encoded in self.quasi_identifer_slices for feature in encoded])
        else:
            all_one_hot_features = set()
            
        for cell in cells:
            cell['representative'] = {}
            # get all rows in cell
            indexes = [index for index, node_id in enumerate(node_ids) if node_id == cell['id']]
            # TODO: should we filter only those with majority label? (using hist)
            rows = x[indexes]
            
            # Check l-diversity & t-closeness
            # Cells that do not meet l-diversity or t-closeness are supressed by using -99 as a placeholder
            # if the sensitive index is numerical and "*" otherwise.
            if self.l and not self._check_l_diversity(rows):
                for idx in self.sensitive_indices:
                    if idx in self.categorical_features:
                        rows[:, idx] = "*"
                    else:
                        rows[:, idx] = -99
                x[indexes] = rows
            
            if self.t and not self._check_t_closeness(rows):
                for idx in self.sensitive_indices:
                    if idx in self.categorical_features:
                        rows[:, idx] = "*"
                    else:
                        rows[:, idx] = -99  
                x[indexes] = rows
                
            done = set()
            for feature in self.quasi_identifiers:  # self.quasi_identifiers are numerical indexes
                if feature not in done:
                    # deal with 1-hot encoded features
                    if feature in all_one_hot_features:
                        # find features that belong together
                        for encoded in self.quasi_identifer_slices:
                            if feature in encoded:
                                values = rows[:, encoded]
                                unique_rows, counts = np.unique(values, axis=0, return_counts=True)
                                rep = unique_rows[np.argmax(counts)]
                                for i, e in enumerate(encoded):
                                    done.add(e)
                                    cell['representative'][e] = rep[i]
                    else:  # rest of features
                        values = rows[:, feature]
                        if self.categorical_features and feature in self.categorical_features:
                            # find most common value
                            cell['representative'][feature] = Counter(values).most_common(1)[0][0]
                        else:
                            # find the mean value (per feature)
                            median = np.median(values)
                            min_value = max(values)
                            min_dist = float("inf")
                            for value in values:
                                # euclidean distance between two floating point values
                                dist = abs(value - median)
                                if dist < min_dist:
                                    min_dist = dist
                                    min_value = value
                            cell['representative'][feature] = min_value

    def _find_sample_nodes(self, samples):
        paths = self._anonymizer.decision_path(samples).toarray()
        node_set = set(self._nodes)
        return [(list(set([i for i, v in enumerate(p) if v == 1]) & node_set))[0] for p in paths]

    def _find_sample_cells(self, samples, cells_by_id):
        node_ids = self._find_sample_nodes(samples)
        return [cells_by_id[node_id] for node_id in node_ids]

    def _anonymize_data(self, x, x_anonymizer_train, cells_by_id):
        cells = self._find_sample_cells(x_anonymizer_train, cells_by_id)
        index = 0
        for row in x:
            cell = cells[index]
            index += 1
            for feature in cell['representative']:
                row[feature] = cell['representative'][feature]
        return x

    def _modify_categorical_features(self, x):
        # prepare data for DT
        used_features = self.features
        if self.train_only_QI:
            used_features = self.quasi_identifiers
        numeric_features = [f for f in self.features if f in used_features and f not in self.categorical_features]
        categorical_features = [f for f in self.categorical_features if f in used_features]
        numeric_transformer = Pipeline(
            steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0))]
        )
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )
        encoded = preprocessor.fit_transform(x)
        return encoded
