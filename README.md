Original repository: https://github.com/IBM/ai-privacy-toolkit

### Assigment 1 - Data Protection Technologies

In this work, I studied the work done in https://research.ibm.com/publications/data-minimization-for-gdpr-compliance-in-machine-learning-models, and added security related features to the corresponding code in the original github repository.

First, I will start by describing the added features, and how they contribute to the security of the data. After that I will describe what was changed in the code.


A toolkit for tools and techniques related to the privacy and compliance of AI models.

The [**anonymization**](apt/anonymization/README.md) module contains methods for anonymizing ML model 
training data, so that when a model is retrained on the anonymized data, the model itself will also be 
considered anonymous. This may help exempt the model from different obligations and restrictions 
set out in data protection regulations such as GDPR, CCPA, etc. 

The [**minimization**](apt/minimization/README.md) module contains methods to help adhere to the data 
minimization principle in GDPR for ML models. It enables to reduce the amount of 
personal data needed to perform predictions with a machine learning model, while still enabling the model
to make accurate predictions. This is done by by removing or generalizing some of the input features.

The [**dataset assessment**](apt/risk/data_assessment/README.md) module implements a tool for privacy assessment of
synthetic datasets that are to be used in AI model training.

Official ai-privacy-toolkit documentation: https://ai-privacy-toolkit.readthedocs.io/en/latest/

Installation: pip install ai-privacy-toolkit

For more information or help using or improving the toolkit, please contact Abigail Goldsteen at abigailt@il.ibm.com, 
or join our Slack channel: https://aip360.mybluemix.net/community.

We welcome new contributors! If you're interested, take a look at our [**contribution guidelines**](https://github.com/IBM/ai-privacy-toolkit/wiki/Contributing).

## References
<a id="1">[1]</a> 
Goldsteen, A. et al (2022). 
Data minimization for GDPR compliance in machine learning models.