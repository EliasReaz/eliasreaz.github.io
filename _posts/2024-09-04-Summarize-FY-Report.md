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
    - [Actions and Result](#overview-actions)
    - [Key Definition](#overview-definition)
- [01. Snapshot of pdf report to be summarized](#data-overview)
- [02. Loading Libraries](#loading-libraries)
- [03. Prompt Template and LLMChain to run queries against LLM](#Prompt-LLMChain)
- [04. Response and Summary](#response)
- [05. Streamlit deployment](#rf-title)
  
___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Imagine we have tons of pdf reports and we need to summarize each of these report within a short period of time. Using human resources and manually summarizing each long report would take hours to complete, edit, and finalize a summary report. Generative AI has a rescue to make hunders of long reports' summary within minutes. As an use-case, we have used a 29 page Nike Earnings Call Conference and converted into a 1 page summary using LangChain.     

### Actions and Result<a name="overview-actions"></a>

Using OpenAI API key connecting to ChatGPT-3.50-turbo AI conversation, we write a python script that loads the 29 page pdf report using LangChain's PyPDFLoader. Utilizing LangChain's PromptTemplate and LLMChains, the summary is provided in numbered bullet points. 

### Key Definition <a name="overview-definition"></a>

Large Language Models (LLMs) are machine learning models that can comprehend and generate human language text.

LangChain is a framework for developing applications powered by Large Language Models (LLMs). 

PyPDFLoader(filepath).load() reads pdf file, splits text by page, index each page in standardized LangChain document structure with "metadata" and "page_content"

# Snapshot of pdf report to be summarized <a name="data-overview"></a>

<image src="/img/posts/NIKE-Inc-Q3FY24-OFFICIAL-Transcript-FINAL.pdf">

# Loading Libraries <a name="loading-libraries"></a>

```python
# LIBRARIES AND SETUP

# langchain community allows third party integration into langchain
# langchain_community document_loader has many document loaders: 
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

# Prompt Template and LLMChain to run queries against LLM <a name="Prompt-LLMChain"></a>

```python
# 2.0 EXPANDING WITH PROMPT TEMPLATES
customize_template = """
Write a concise summary on this {text}

Give the summary in 4 to 7 numbered bullet points
"""
# Prompt template for a language model
doc_prompt = PromptTemplate(input_variables=["text"], 
                        template = customize_template
                        )
# Chain to run queries against LLMs.
llm_chain = LLMChain(llm=model, prompt=doc_prompt)


```









