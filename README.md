# Assignment 1 - Data Protection Technologies  

In this work, I studied the research presented in [Data Minimization for GDPR Compliance in Machine Learning Models](https://research.ibm.com/publications/data-minimization-for-gdpr-compliance-in-machine-learning-models) and implemented security-related enhancements in the corresponding code from the original [GitHub repository](https://github.com/IBM/ai-privacy-toolkit).  

First, I will describe the newly added features and how they contribute to data security. Then, I will outline the modifications made to the original code.  

---

## Added Features  

### l-Diversity  

The original work implements **k-anonymity** to protect individual privacy by ensuring that each record in the dataset is indistinguishable from at least **k-1** other records.  

However, k-anonymity alone is susceptible to attacks such as the **homogeneity attack** and the **background knowledge attack**.  

**l-Diversity** extends k-anonymity by ensuring that each equivalence class contains at least **l distinct values** for the sensitive attribute. This prevents attackers from inferring sensitive information, even if they recognize a group of anonymized records.  

To enforce l-diversity, I implemented **suppression**, which is simpler to apply than generalization. While suppression results in greater information loss, it provides a higher level of privacy compared to generalization.  

---

### t-Closeness  

**t-Closeness** is a further refinement of **l-Diversity** that improves privacy by ensuring that the distribution of sensitive attributes in each equivalence class is close to the overall distribution in the dataset.  

A dataset satisfies **t-closeness** if the distance between the distribution of a sensitive attribute in each equivalence class and its distribution in the entire dataset does not exceed a predefined threshold **t**.  

To measure this distance, I used the **Earth Mover’s Distance (EMD)**, following the original paper on t-closeness ([Reference](https://www.cs.purdue.edu/homes/ninghui/papers/t_closeness_icde07.pdf)).  

The **t parameter (0 to 1)** allows a trade-off between data utility and privacy, with **lower values of t** enforcing stricter privacy constraints.  

---

## Modifications in the Original Code  

The following files were modified:  

- **`ai-privacy-toolkit/apt/anonymization/anonymizer.py`**  
  - Added parameters: **l, t, and sensitive_attributes** to the `Anonymize` class.  
  - Added instance attributes: **self.l, self.t, self.sensitive_attributes, self.global_distribution**.  
  - Implemented new methods:  
    - `_check_l_diversity`  
    - `_calculate_global_distribution`  
    - `_check_t_closeness`  
  - Modified existing methods:  
    - `_find_representatives` (**lines 243-260**)  
    - `anonymize` (**lines 109-115**)  

- **`ai-privacy-toolkit/notebooks`**  
  - Updated the following notebooks to evaluate the impact of the new features:  
    - `anonymization_one_hot_adult.ipynb`  
    - `attribute_inference_anonymization_nursery.ipynb`  

- **`ai-privacy-toolkit/apt/utils/dataset_utils.py`**  
  - Modified dataset loading functions:  
    - `get_adult_dataset_pd` (**lines 143-145**)  
    - `get_nursery_dataset_pd` (**lines 251-252**)  

---

## How to Run the Code  

All modifications can be tested by running the updated notebooks:  
- **`anonymization_one_hot_adult.ipynb`**  
- **`attribute_inference_anonymization_nursery.ipynb`**  

---
## Results  

The results indicate that both **l-diversity** and **t-closeness** effectively enhance data privacy during model training by reducing the granularity of data representation. Furthermore, they provide **better protection when used together**.  

However, these privacy-preserving techniques **reduce model accuracy**. Therefore, it is crucial to define an appropriate privacy-utility trade-off, ensuring that the decrease in accuracy remains within an acceptable threshold.  
