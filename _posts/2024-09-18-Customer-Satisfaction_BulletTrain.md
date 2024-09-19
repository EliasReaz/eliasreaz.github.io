---
layout: post
title: Passengers' satisfaction in High Speed Trains - Investigating Key Features 
image: "/posts/bullet_train.png"
tags: [Random Forest, XGBoost, Feature Importance, Python]
---

This problem statement is based on the Shinkansen Bullet Train in Japan, and passengers’ experience with that mode of travel. This machine-learning
exercise aims to determine the relative importance of each parameter with regard to their contribution to the passengers’ overall travel experience. 

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions and Result](#overview-actions)
    - [Key Definition](#overview-definition)
- [01. Dataset Snapshot](#dataset-snapshot)
- [02. Loading Libraries](#loading-libraries)
- [03. Exploratory Data Analysis](#eda)
- [04. Random Forest Classifier](#rf)
- [05. XGBoost Classifier](#xgboost)
- [06. Feature importance](#fi)
- [07. Summary](#summary)

  
___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

This problem statement is based on the Shinkansen Bullet Train in Japan, and passengers’ experience with that mode of travel. This machine-learning
exercise aims to determine the relative importance of each parameter with regard to their contribution to the passengers’ overall travel experience. The
dataset contains a random sample of individuals who travelled on this train. The on-time performance of the trains along with passenger information is
published in a file named ‘Traveldata_train.csv’. These passengers were later asked to provide their feedback on various parameters related to the
travel along with their overall experience. These collected details are made available in the survey report labelled ‘Surveydata_train.csv’.  

The objective of this problem is to understand which parameters play an important role in swaying passenger feedback towards a positive scale.

### Actions and Result<a name="overview-actions"></a>

- The target feature is 'Overall Experience' as 0 (unsatisfied) and 1 (satisfied)
- The number of input features are 24. These are 'Gender', 'Customer_Type', 'Age', 'Type_Travel', 'Travel_Class', 'Travel_Distance', 'Departure_Delay_in_Mins', 'Arrival_Delay_in_Mins', 'Overall_Experience', 'Seat_Comfort', 'Seat_Class', 'Arrival_Time_Convenient', 'Catering', 'Platform_Location', 'Onboard_Wifi_Service', 'Onboard_Entertainment', 'Online_Support', 'Ease_of_Online_Booking', 'Onboard_Service', 'Legroom', 'Baggage_Handling', 'CheckIn_Service', 'Cleanliness', 'Online_Boarding'.
- The number of observations used as a training set is around 94,379.
- Two Classification models are used: Random Forest and XGBoost

### Key Definition <a name="overview-definition"></a>

- Random Forest: An ensemble of multiple decision trees. Each tree is constructed using a random subset of the dataset and a random subset of features. The results of each tree is aggregated, for classification, by majority voting.

- XGBoost (eXtream Gradient Boosting): An ensemble model that builds trees sequentially, each new tree improves by learning from mistakes of the previous ones. Boosting combines multiple individual weak trees. After hundreads of iterations, weak learners are converted to strong learners. It is a supervised learning boosting algorithm that uses gradient descent.    


# Dataset Snapshot <a name="dataset-snapshot"></a>

Travel data:


| ID |Gender |	Customer_Type |	Age	| Type_Travel |	Travel_Class | Travel_Distance | Departure_Delay_in_Mins | Arrival_Delay_in_Mins |
| --- | --- |	--- |	---	| --- | --- | --- | --- | --- |
| 98800001 | Female	| Loyal Customer| 52.0| 	| Business | 272 |	0.0 |	5.0 |
|98800002 |	Male | Loyal Customer	 |48.0	|Personal Travel |Eco|	2200|	9.0|	0.0|
|98800003 | Female |Loyal Customer |43.0 |	Business Travel|	Business|	1061|	77.0|	119.0|

Passenger survey data:

| ID	| Overall_Experience | Seat_Comfort | Seat_Class | Arrival_Time_Convenient | Catering | Platform_Location | Onboard_Wifi_Service | Onboard_Entertainment | Online_Support | Ease_of_Online_Booking | Onboard_Service | Legroom |	Baggage_Handling |	CheckIn_Service |	Cleanliness	| Online_Boarding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |	--- |	--- |	---	| --- | 
| 98800001 | 0 | Needs Improvement | Green Car | Excellent | Excellent | Very Convenient | Good | Needs Improvement | Acceptable | Needs Improvement | Needs Improvement | Acceptable |	Needs Improvement |	Good | Needs  Improvement |Poor|
|98800002 |	1 | Acceptable | Ordinary |	Acceptable | Acceptable | Manageable | Needs Improvement |	Good | Excellent | Good | Good | Good |	Good |	Good |	Good |	Good |


# Loading Libraries <a name="loading-libraries"></a>

```python
# Basic libraries of python for numeric and dataframe computations
import numpy as np
import pandas as pd
import scipy.stats as stats
# Basic library for data visualization
import matplotlib.pyplot as plt
# Slightly advanced library for data visualization
import seaborn as sns
# import sklearn libraries
from sklearn.utils import shuffle
from sklearn.model_selection import (
                train_test_split, 
                cross_val_score, 
                KFold)
from sklearn.metrics import (
            confusion_matrix, 
            accuracy_score, 
            precision_score, 
            recall_score, 
            f1_score, classification_report)
from sklearn.preprocessing import OneHotEncoder
# RandomForestCLassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
# Instead of randomly testing different optimizationoptions, 
# BayesSearchCV focuses on most promising areas which saves times and gives better result
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.inspection import permutation_importance
# Used to ignore the warning given as output of the code
import warnings
warnings.filterwarnings("ignore")
```

# Exploratory Data Analysis <a name="eda"> </a>

- By Gender, Female travellers are 51% and Male are 49%.
- Female travellers showed more satisfied experience than male travellers.
- By Travel type, Business Travellers are 69% while Personal Travellers are 31%.
- Business Travellers showed higher satisfied experience. 
- By Travel Class, Economy class travellers are 52% and Business are 48%.
- By loyalty, Loyal Customers are 82%.

<br>

![Gender](/img/posts/Screenshot_gender.png)

<br>

<br>

![Gender](/img/posts/Screenshot_customertype.png)

<br>

<br>

![Gender](/img/posts/Screenshot_onboard_entertainment.png)

<br>



# Random Forest Classifier<a name="rf"></a>

```python
# Choose the type of classifier 
rf_tuned = RandomForestClassifier(random_state = 42)

# Grid of parameters to choose from
parameters = {"n_estimators": [200, 300, 500, 600],
    "max_depth": [4, 5, 6],
    "min_samples_leaf": [20, 25],
    "min_samples_split": [3, 5],
    'max_features': [5, 7],
    "criterion": ["entropy"],
      }
# Run the grid search on the training data using cv=5
grid_obj = RandomizedSearchCV(estimator=rf_tuned, 
                              param_distributions = parameters, 
                              cv = 5, 
                              n_iter=10,
                              scoring='accuracy',  
                              verbose=3, 
                              # error_score="raise"
                             )

grid_obj = grid_obj.fit(X_train, y_train)
best_param_rf = grid_obj.best_params_
# Instantiate the classifier with the best parameter
random_forest = RandomForestClassifier(**best_param_rf)
# Fit with train data
random_forest.fit(X_train, y_train)
# Predict
y_pred_rf_class = random_forest.predict(X_test)
print("Random Forest Classifier with GridSearchCV: Classification Report (Test set)")
print("*"*50)
print(classification_report(y_true = y_test, y_pred = y_pred_rf_class, digits=4))
```

<br>

The Random Forest Classifier shows 90% accuracy on the test dataset. 

| Overall Experience      | precision | recall | f1-score | support |
| ---   | --- | --- | --- | --- |         
|  0 (unsatisfied)   | 0.8946 |   0.8841|    0.8893|      8557|
|  1  (satisfied) |  0.9048 |    0.9137 |    0.9092 |     10319 |
| Accuracy |         |          |      0.9002 |     18876 |


# XGBoost Classifier <a name="xgboost"></a>

```python
param_dist = {
    'max_depth': stats.randint(3, 15),
    'learning_rate': stats.uniform(0.01, 0.3),
    'subsample': stats.uniform(0.6, 0.4),
    'n_estimators': stats.randint(50, 200),
    'gamma': stats.uniform(0, 0.5),
    'colsample_bytree': stats.uniform(0.5, 0.5),
    'min_child_weight': stats.randint(1, 10),
    'scale_pos_weight': stats.uniform(1, 10)
}


# Create the XGBoost model object
xgb_model = xgb.XGBClassifier()

# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(
                    estimator = xgb_model, 
                    param_distributions = param_dist, 
                    n_iter=50, 
                    cv=10, 
                    scoring='accuracy',
                    n_jobs = 1, # use all resources, i.e., all available cores
                    verbose=3, 
                    )

# Fit the RandomizedSearchCV object to the training data
random_search.fit(X_train, y_train)
# Print the best set of hyperparameters and the corresponding score
print("Best set of hyperparameters: ", random_search.best_params_)
print("Best score: ", random_search.best_score_)

b_param  = random_search.best_params_

xgb_model_opt1 = xgb.XGBClassifier(**b_param)

xgb_model_opt1.fit(X_train, y_train)

y_pred_class1 = xgb_model_opt1.predict(X_test)
```

<br>


XGBoost provides 95% accuracy on the test dataset.

| Overall Experience      | precision | recall | f1-score | support |
| ---   | --- | --- | --- | --- |         
|  0 (unsatisfied)   | 0.9508 | 0.9371 |  0.9439|      8557|
|  1  (satisfied) |  0.9485 |    0.9598 |    0.9541 |     10319 |
| Accuracy |         |          |      0.9495 |     18876 |


# Feature Importance <a name="fi"></a>

<br>


![FeatureImportance](/img/posts/Screenshot_feature_importance.png)



# Summary<a name="summary"></a>

- The project predicts customer satisfaction as 0 (unsatisfied) or 1 (satisfied). The customer is a Japanese bullet train service. The number of input features are 24. These are 'Gender', 'Customer_Type', 'Age', 'Type_Travel', 'Travel_Class', 'Travel_Distance', 'Departure_Delay_in_Mins', 'Arrival_Delay_in_Mins', 'Overall_Experience', 'Seat_Comfort', 'Seat_Class', 'Arrival_Time_Convenient', 'Catering', 'Platform_Location', 'Onboard_Wifi_Service', 'Onboard_Entertainment', 'Online_Support', 'Ease_of_Online_Booking', 'Onboard_Service', 'Legroom', 'Baggage_Handling', 'CheckIn_Service', 'Cleanliness', 'Online_Boarding'. The number of observations used for training is around 94.3 K.

- Two Classification models are used: Random Forest and XGBoost. For both of these models RandomizedSearchCV is used to find suitable hyperparamerts.

- Based on accuracy, Random Forest gives 90% accuracy, while XGBoost gives 95% accuracy. Therefore, XGBoost is selected to use for the prediction of overall passenger experience.

- Most influencial features that drive customer satisfaction are Seat Comfort, Onboard Entertainment, Checkin Service, Baggage Handling, Online support, Cleanliness.







