---
layout: post
title: Predicting Customer Loyalty Using ML
image: "/posts/regression-title-img.png"
tags: [Customer Loyalty, Machine Learning, Regression, Python]
---

Our client, a grocery retailer, hired a market research consultancy to append market level customer loyalty information to the database.  However, only around 50% of the client's customer base could be tagged, thus the other half did not have this information present.  Let's use ML to solve this!

## Table of contents

- [Table of contents](#table-of-contents)
- [Executive Summar](#executive-summary)
  - [Context](#context)
  - [Objective](#objective)
  - [Results and Summary](#results-summary)
  - [Growth/Next Step](#growth-next-steps)
- [Technical Details](#technical-details)
  - [Data Overview](#data-overview)
  
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
|---      | ---                 | --- |
|Random Forest | 0.955 | 0.925|
|Decision Tree | 0.886 | 0.871|
|Linear Regression | 0.754 | 0.853|

<br>
<br>

![alt text](/img/post/rf-regression-permutation-importance.png "Random Forest Permutation Importance Plot")

### Growth/Next Steps <a name="growth-next-steps"></a>

- Monitor model performance when more customer data are available.
- Employ ML model like XGBoost and see how the model performs.
- Study more customer traits like house income, age, job/business types if available in future and make new features.


<br>
<br>


## Technical Details  <a name="technical-details"></a>

### Data Overview <a name="data-overview"></a>

The retailer has customer details, their transactions history, and loyalty score for the half of customers.

|TABLE NAME | Field name |Field name     | Field name        |   Field name     |
|---        | ---             |--           |---      |---     |
|transactions|  customer_id|trasaction_date | product_name_id | total_sales|
|customer_details|customer_id|distance_from_shop|gender|
|customer_loyalty_scores|customer_id|loyalty_score| | 
