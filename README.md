Original repository: https://github.com/IBM/ai-privacy-toolkit

### Assigment 1 - Data Protection Technologies

In this work, I studied the work done in https://research.ibm.com/publications/data-minimization-for-gdpr-compliance-in-machine-learning-models, and added security related features to the corresponding code in the original github repository.

First, I will start by describing the added features, and how they contribute to the security of the data. After that, I will describe what was changed in the code.

##### Added Features

- L-diversity:
In the original work, k-anonymity is used in order to protect the privacy of individuals in the data by ensuring that each record in the data is indistinguishable from at least k-1 other records.

While k-anonymity is a great approach, it is still susceptible to many attacks (eg. homogeinity attack and backgroud knowledge atack).

The l-diversity is an extension of the k-anonymity that addresses some of the weaknesses of the k-anonymity. It does that by ensuring that within each group there are at least l different values for the sensitive attribute. We can achieve that by using techniques such as generalization and supression. In this work I used suppresion, since it is simple to implement when comparing to the generalization. Despite the fact that we lose more information with suppresion, we are able to achieve an higher level of privacy when comparing to generalization.