---
layout: post
title: Targeting Smarter: Predicting Customer Signups Using Machine Learning
image: "/posts/classification-title-img.png"
tags: [Customer Targeting, Machine Learning, Classification, Python]
---

A grocery retailer sought to reduce high mailing costs by targeting only the customers most likely to sign up for their *delivery club* membership program. Using machine learning classification models, we predicted sign-up probability, optimized campaign targeting, and significantly improved ROI.

# Table of Contents

- [Project Overview](#project-overview)
- [Data Overview](#data-overview)
- [Modelling Techniques](#modelling-techniques)
- [Model Comparison](#model-comparison)
- [Application & Business Value](#application--business-value)
- [Growth & Next Steps](#growth--next-steps)

---

# Project Overview

## Context

In a previous campaign, the client mailed their entire customer base (except a control group) to promote a $100/year *delivery club* that offered free grocery deliveries (vs. $10/delivery otherwise). Only 31% of customers signed up, making the campaign expensive and inefficient.

To optimize future campaigns, the goal was to build a predictive model that identifies which customers are most likely to subscribe.

## Actions

- Combined data from multiple sources: transaction history, demographics, and product categories  
- Engineered features reflecting purchase behavior and customer-store proximity  
- Built and evaluated four classification models:  
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - K-Nearest Neighbors (KNN)  
- Assessed performance using **Accuracy**, **Precision**, **Recall**, and **F1 Score**  
- Selected the best model based on both predictive accuracy and interpretability

## Results

The **Random Forest** model was selected for deployment based on its superior performance and explainability.

### Final Performance Metrics

| Model                | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| **Random Forest**     | 0.935    | 0.887     | 0.904  | 0.895    |
| Decision Tree        | 0.929    | 0.885     | 0.885  | 0.885    |
| KNN                  | 0.936    | 1.000     | 0.762  | 0.865    |
| Logistic Regression  | 0.866    | 0.784     | 0.690  | 0.734    |

- **Top predictive features:**  
  - `distance_from_store`  
  - `transaction_count`

---

# Data Overview

The dataset was prepared by merging customer-level data from the following sources:

- `transactions` table: purchase history  
- `customer_details` table: demographics and credit score  
- `product_areas` table: variety of products purchased  

We used a 3-month lookback window prior to the original campaign to generate the predictors.

### Final Dataset Sample

| Variable               | Type        | Description |
|------------------------|-------------|-------------|
| `signup_flag`          | Target      | Whether the customer signed up for the delivery club |
| `distance_from_store`  | Continuous  | Distance in miles between customer and store |
| `gender`               | Categorical | M, F, or Unknown |
| `credit_score`         | Continuous  | Recent credit score |
| `total_sales`          | Continuous  | Total spend in past 3 months |
| `total_items`          | Continuous  | Total items purchased |
| `transaction_count`    | Continuous  | Total number of transactions |
| `product_area_count`   | Discrete    | Number of distinct product areas purchased from |
| `average_basket_value` | Continuous  | Avg spend per transaction |

---

# Modelling Techniques

Each model followed this general workflow:

- Data cleaning (handling missing values, encoding categorical variables)
- Feature selection (e.g., using RFECV for Logistic Regression and KNN)
- Model training and evaluation on train/test split (80/20 stratified)
- Performance analysis using multiple metrics (not just accuracy)

### Special Preprocessing Considerations

- **Outlier removal** for distance and spend-related fields (especially for Logistic Regression and KNN)
- **Feature scaling** (MinMaxScaler) was applied only for KNN
- **Categorical encoding** using One-Hot Encoding for all models

---

# Model Comparison

| Model               | Best Metric Strength                   | Weakness                                  |
|--------------------|----------------------------------------|--------------------------------------------|
| **Random Forest**     | Best F1 Score (0.895); high recall    | More complex to interpret than Decision Tree |
| Decision Tree       | Interpretable + strong recall (0.885) | Slightly lower precision |
| KNN                 | Perfect precision (1.00)               | Low recall (0.762), which penalized F1     |
| Logistic Regression | Fast and simple                        | Weak recall (0.690), lower overall metrics  |

---

# Application & Business Value

With the final model (Random Forest), we can now:

- Score future customers with sign-up probabilities
- Recommend **targeted mailing** to only high-likelihood customers
- Reduce mailing cost significantly while maintaining or increasing conversion
- Use feature importance insights to tailor messaging (e.g., proximity-based offers)

---

# Growth & Next Steps

### Model Improvements
- Test gradient boosting models: **XGBoost**, **LightGBM**, and **CatBoost**
- Tune hyperparameters (e.g., tree depth, number of trees)
- Consider ensemble stacking for improved robustness

### Data Enhancements
- Add customer tenure, income bands, and digital interaction data
- Introduce time-based features (e.g., recency, frequency)

### Production Planning
- Create pipeline for automated scoring during each campaign cycle
- Integrate with marketing automation system for seamless deployment

---

*Thanks for reading! If you’re interested in customer segmentation, predictive analytics, or campaign optimization, feel free to [connect with me on LinkedIn](https://www.linkedin.com/in/reaz-ussalamelias/) or check out my notes on [my notes](https://eliasreaz.github.io/datascience-notes).*
