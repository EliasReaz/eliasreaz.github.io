---
layout: post
title: Leveraging Gen AI for Summarization of Large Volume of Text
image: "/posts/LangChain_logo.png"
tags: [Generative AI, OpenAI, LangChain, Python]
---

We have converted a 29 page Earnings Call Transcript from Nike to a 1 page bullet point summary leveraging Generative AI.  

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Key Definition](#overview-definition)
- [01. Sample of Report](#data-overview)
- [02. Loading Libraries](#modelling-overview)
- [03. Prompt Template and chain to run queries against LLM](#linreg-title)
- [04. Response and Summary](#regtree-title)
- [05. Streamlit deployment](#rf-title)
___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Imagine we have tons of pdf reports and we need to summarize each of these report within a short period of time. Using human resources would take hours to make reports. Generative AI has come to help to make summarized reports within minutes. As an use-case, we have used a 29 page Nike Earnings Call Conference and converted into a 1 page summary using LangChain.     

(in progress)
<br>
<br>

### Actions <a name="overview-actions"></a>

We have opened an account in OpenAI and have OpenAI API Key. We have a python script that load the 29 page pdf report using PyPDFLoader from LangChain. Connecting via OpenAI API key we invoke LangChain utilities to summarize a nice report. 
