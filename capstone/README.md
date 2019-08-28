# Machine Learning Engineer Nanodegree

## Capstone Project: Predicting customer churn

Implements a machine learning classifier to predict customer churn in the telco industry based on customer data. Read the full report in [Capstone Churn Prediction Report Manuel Seeger.pdf](Capstone Churn Prediction Report Manuel Seeger.pdf)
### Implementation

Run capstone.ipynb to reproduce the results of the project.

At the beginning of the notebook there are a number of global parameters that control performance intensive tasks. Adjust depending on your host machine. 

### Data

Required dataset is included in subfolder data in cell2celltrain.csv. 

### Dependencies

* Python 3.5 or greater
* Required Python libraries are stated in requirements.txt
* Subfolder global_objectives needs to be in the python path of the main notebook to load the custom loss layers. This code is a copy of the Tensorflow reserch module global_objectives.
* utils.py contains convenience functions for the main notebook
* The project uses tensorflow modules that won't be included in TF 2.0. Run with TF 1.x

No further dependencies required to run the Notebook. 

requirements.txt
````
scikit-learn
pandas
numpy
keras
xgboost
tensorflow
matplotlib
imbalanced-learn
scipy
````