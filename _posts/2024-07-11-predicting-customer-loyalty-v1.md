---
layout: post
title: Predicting Customer Loyalty Using ML
image: "/posts/regression-title-img.png"
tags: [Customer Loyalty, Machine Learning, Regression, Python]
---

Our client, a grocery retailer, hired a market research consultancy to append market level customer loyalty information to the database.  However, only around 50% of the client's customer base could be tagged, thus the other half did not have this information present.  Let's use ML to solve this!

## Table of contents
- [Table of contents](#table-of-contents)
- [Executive Summary](#executive-summary-)
  - [Context](#context-)
  - [Objective](#objective-)
  - [Results and Summary](#results-and-summary-)
  - [Growth/Next Steps](#growthnext-steps-)
- [Technical Details](#technical-details--)
  - [Data Overview](#data-overview-)
    - [customer\_details Table](#customer_details-table)
    - [transactions Table](#transactions-table)
    - [product\_areas Table:](#product_areas-table)
    - [loyalty\_scores Table:](#loyalty_scores-table)
  - [Data Processing](#data-processing)
  
---

## Executive Summary <a name="executive-summary"></a>

### Context <a name="context"></a>

Customers who trust  retailer's products and find the products benecifical are expected to visit the shop again and will buy products/offers more. We tag these customers as loyal customers and we assign a measure called **loyalty score** to customers. By loyalty score we refer to the % of grocery spend (market level) that each customer allocates in our client retailer against all of the retailer's competitors. If customer A has total grocery budget of 100 dollars and she spends 80 dollars with our client, her loyalty score is 0.80. If customer B has total grocery budget of 200 dollars and he spends 35% with our client and 65% with all competitors, his loyalty score is 0.35.

### Objective <a name="objective"></a>

Currently the retailer database has loyalty score for 50% of customers. **This project aims to accurately estimate loyalty score for the rest of customers** based on the pattern or relationship of customer traits/ features that are available in the database. These traits/ features (we also say input features) are the customer proximity around the retailer shop, total expenditure, number of product types, total number of items, customer credit score, and gender.


### Results and Summary <a name="results-summary"></a>

- We have utilized three machine learning (ML) approaches to accurately predict customer loyalty scores based on customer demographics and behavioral traits.
- These ML models are: Linear Regression, Decision Tree, and Random Forests.
- Results show that the ML model Random Forests outperforms the other two models with 93% accuracy.
- To understand which customer traits influence the loyalty score, we have investigated relative feature importance and found that customers who live near the retailer shop are more loyal than who shop from far. For details, please see the following table and figure.

|ML Model | Adjusted r-squared | r-squared (Cross Validation K=4) |
|---      | :---:              | :---: |
|Random Forest | 0.955 | 0.925|
|Decision Tree | 0.886 | 0.871|
|Linear Regression | 0.754 | 0.853|

<br>

![alt text](/img/posts/rf-regression-permutation-importance.png "Random Forest Permutation Importance Plot")

### Growth/Next Steps <a name="growth-next-steps"></a>

- Monitor model performance when more customer data are available.
- Employ ML model like XGBoost and see how the model performs.
- Study more customer traits like house income, age, job/business types if available in future and make new features.

<br>

## Technical Details  <a name="technical-details"></a>

### Data Overview <a name="data-overview"></a>

The retailer has customer details, their transactions history, and loyalty score for the half of customers.

|TABLE NAME | Field name |Field name     | Field name        |   Field name  |   Field name |
|---        | ---             |--           |---      |---     | ---|
|customer_details|customer_id|distance_from_shop|gender| credit_score|
|transactions|  customer_id|transaction_date | num_items | product_area_id |sales_cost |
|customer_loyalty_scores|customer_id|loyalty_score| | | |

#### customer_details Table

| customer_id |  distance_from_store | gender | credit_score |
| ------       | -------             |---    | ---         |  
|754          | 1.17                 | M      | 0.75       |
|865          | 0.17                 | F      | 0.35       |
|954          |                      | M      |           |

#### transactions Table

| customer_id |  transaction_date | transaction_id | product_area_id | num_items | sales_cost |
| ----        | ----             |-------------    | -------       | -------   | -------  |
|647          | 2020-04-01       | 4355233          |   4         | 3           |9.33      |
|647          | 2020-04-01       | 4355233          |   3         | 4           |23.82      |
|439          | 2020-07-15       | 4355500          |   4         | 1           |6.83      |

#### loyalty_scores Table:

|customer_id | customer_loyalty_score |
| -----      | ----------             |
| 104        | 0.587                  |
| 69        | 0.156                  |
| 796        | 0.428                  |

### Data Processing <a name="data-processing"></a>
We find that data are scattered in three different tables - customer details, loyalty scores, and transactons. So we did some feature engneering as follows:

- merge customer_details with loyalty_scores table on customer id, so that we have loyalty scores (both non null and null) along with distance_from_shop, gender, and credit_score. We name this dataframe as grocery_data.

```python
import pandas as pd
import pickle

loyalty_score = pd.read_excel("data/grocery_database.xlsx", sheet_name="loyalty_scores")
customer_details = pd.read_excel("data/grocery_database.xlsx", sheet_name="customer_details")
transactions = pd.read_excel("data/grocery_database.xlsx", sheet_name="transactions")

##########################################
# merge customer_details and loyalty_score
###########################################
grocery_data = pd.merge(left=customer_details, right=loyalty_score, 
                        how="left", on="customer_id")
```

- make new features from transactions table grouping by customer_id for aggregating each customer's total_cost, total_num_items, transaction_count, and product area count. We name this dataframe as sales_summary.
  
<br>

```python
##########################################
# groupby customers from transactions aggregating total sales cost, total number of items, count of transaction, and product area id 
###########################################

sales_summary = transactions.groupby("customer_id").agg({"sales_cost":"sum", 
                                        "num_items":"sum", 
                                        "transaction_id":"count", 
                                        "product_area_id":"nunique"}).reset_index()
sales_summary.columns = ["customer_id", "total_sales", "total_items", 
                         "transaction_count", "product_area_count"]

```
<br>

- merge dataframes, grocery_data and sales_summary

<br>

```python
################################################
## merge sales_summary with grocery_data
################################################
data_for_regression= pd.merge(grocery_data, sales_summary, how="inner", on="customer_id")
print(data_for_regression.tail())

```

<br>

- make a feature named average_busket_value assuming that customers with higher average value per transaction will be more loyal.

<br>

```python
## Assumption is that customers with higher mean value per transaction is more loyal 
sales_summary["average_basket_value"] = sales_summary["total_sales"]/sales_summary["transaction_count"]

```

<br>

- make two dataframes - one dataset for modeling having loyalty scores and the other dataset for predicting with no layalty scores.
- save two dataframes as pickel files for future use,


```python
####################################################
## data for model building  having loyalty scores are used 
#####################################################
regression_modeling = data_for_regression.loc[data_for_regression["customer_loyalty_score"].notna()]
###############################################
## data for prediction  with no loyalty scores 
##############################################
regression_scoring = data_for_regression.loc[data_for_regression["customer_loyalty_score"].isna()]

regression_scoring.drop(["customer_loyalty_score"], axis=1, inplace=True)
#######################################################
## save as pickle files
###################################################### 
pickle.dump(regression_modeling, open("data/regression_modeling.p", "wb"))
pickle.dump(regression_scoring, open("data/regression_scoring.p", "wb"))
```

<br>
After this data pre-processing in Python, we have a dataset for modelling that contains the following fields...
<br>
<br>

| **Variable Name** | **Variable Type** | **Description** |
|---|---|---|
| **loyalty_score** | **Dependent** | The % of total grocery spend that each customer allocates to ABC Grocery vs. competitors |
| distance_from_store | Independent | "The distance in miles from the customers home address, and the store" |
| gender | Independent | The gender provided by the customer |
| credit_score | Independent | The customers most recent credit score |
| total_sales | Independent | Total spend by the customer in ABC Grocery within the latest 6 months |
| total_items | Independent | Total products purchased by the customer in ABC Grocery within the latest 6 months |
| transaction_count | Independent | Total unique transactions made by the customer in ABC Grocery within the latest 6 months |
| product_area_count | Independent | The number of product areas within ABC Grocery the customers has shopped into within the latest 6 months |
| average_basket_value | Independent | The average spend per transaction for the customer in ABC Grocery within the latest 6 months |






