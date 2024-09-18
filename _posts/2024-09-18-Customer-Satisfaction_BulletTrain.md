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
- [01. Loading Libraries](#loading-libraries)
- [02. Load document and initialize model](#load-doc-initialize-model)
- [03. Prompt Template and LLMChain to run queries against LLM](#Prompt-LLMChain)
- [04. Response](#response)
- [05. Summary](#summary)
  
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

travel_data:


| ID |Gender |	Customer_Type |	Age	| Type_Travel |	Travel_Class | Travel_Distance | Departure_Delay_in_Mins | Arrival_Delay_in_Mins |
| --- | --- |	--- |	---	| --- | --- | --- | --- | --- |
| 98800001 | Female	| Loyal Customer| 52.0| 	| Business | 272 |	0.0 |	5.0 |
|98800002 |	Male | Loyal Customer	 |48.0	|Personal Travel |Eco|	2200|	9.0|	0.0|
|98800003 | Female |Loyal Customer |43.0 |	Business Travel|	Business|	1061|	77.0|	119.0|

survey data

| ID	| Overall_Experience | Seat_Comfort | Seat_Class | Arrival_Time_Convenient | Catering | Platform_Location | Onboard_Wifi_Service | Onboard_Entertainment | Online_Support | Ease_of_Online_Booking | Onboard_Service | Legroom |	Baggage_Handling |	CheckIn_Service |	Cleanliness	| Online_Boarding |


# Loading Libraries <a name="loading-libraries"></a>

```python
# Basic libraries of python for numeric and dataframe computations
import numpy as np
import pandas as pd

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

# Used to ignore the warning given as output of the code
import warnings
warnings.filterwarnings("ignore")

```
