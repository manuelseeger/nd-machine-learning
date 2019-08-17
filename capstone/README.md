# Machine Learning Engineer Nanodegree

## Capstone Project: Predicting customer churn

### Implementation

Run capstone.ipynb to reproduce the results of the project.

At the beginning of the notebook there are a number of global parameters that control performance intensive tasks. Adjust depending on your host machine. 

### Data

Required dataset is included in subfolder data in cell2celltrain.csv. 

### Dependencies

* Required Python libraries are stated in requirements.txt
* Subfolder global_objectives needs to be in the python path of the main notebook to load the custom loss layers.
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