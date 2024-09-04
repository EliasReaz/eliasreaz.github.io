---
layout: post
title: Summarizing report using OpenAI and LangChain
image: "/posts/LangChain_logo.png"
tags: [Generative AI, OpenAI, LangChain, Python]
---


We summarize a 29 page Nike financial report using OpenAI and LangChain.   
# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
    - [Key Definition](#overview-definition)
- [01. Data Overview](#data-overview)
- [02. Modelling Overview](#modelling-overview)
- [03. Linear Regression](#linreg-title)
- [04. Decision Tree](#regtree-title)
- [05. Random Forest](#rf-title)
- [06. Modelling Summary](#modelling-summary)
- [07. Predicting Missing Loyalty Scores](#modelling-predictions)
- [08. Growth & Next Steps](#growth-next-steps)

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Summarize bullet format a long Nike financial conversation report.

<br>
<br>
### Actions <a name="overview-actions"></a>

We firstly needed to compile the necessary data from tables in the database, gathering key customer metrics that may help predict *loyalty score*, appending on the dependent variable, and separating out those who did and did not have this dependent variable present.

As we are predicting a numeric output, we tested three regression modelling approaches, namely:
