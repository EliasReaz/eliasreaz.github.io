---
layout: post
title: Passengers' satisfaction in High Speed Trains - Investigating Key Features 
image: "/posts/LangChain_logo.png"
tags: [Random Forest, XGBoost, Feature Importance, Python]
---

This problem statement is based on the Shinkansen Bullet Train in Japan, and passengers’ experience with that mode of travel. This machine-learning
exercise aims to determine the relative importance of each parameter with regard to their contribution to the passengers’ overall travel experience. 

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions and Result](#overview-actions)
    - [Key Definition](#overview-definition)
- [01. Report Snapshot](#snapshot-report)
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

- Large Language Models (LLMs) are machine learning models that can comprehend and generate human language text.

- LangChain is a framework for developing applications powered by Large Language Models (LLMs). 

- PyPDFLoader(filepath).load() reads pdf file, splits text by page, index each page in standardized LangChain document structure with "metadata" and "page_content"

# Report Snapshot <a name="snapshot-report"></a>

Following is a sanpshot of the Nike Earnings Call Conference Report. The Report is 29 page long. We summarize this report into 1 page seven bullet points using Open AI and LangChain.

<br>

![PDF Report](/img/posts/Screenshot_Nike.png)





# Loading Libraries <a name="loading-libraries"></a>

```python
# LIBRARIES AND SETUP

# langchain_community allows third party integration into langchain
# langchain_community document_loader has many document loaders, e.g.,
# google sheets, excel, pdf, csv, email, images, Microsoft suite, and many more. 

from langchain_community.document_loaders import PyPDFLoader

# chatOpenAi handles API calls to OpenAI's chat completions via LangChain Standradized LLM Framework
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain

from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain

import yaml
from pprint import pprint

```
