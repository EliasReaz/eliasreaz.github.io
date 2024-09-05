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
- [01. Report Snapshot](#snapshot-report)
- [01. Loading Libraries](#loading-libraries)
- [02. Load document and initialize model](#load-doc-initialize-model)
- [03. Prompt Template and LLMChain to run queries against LLM](#Prompt-LLMChain)
- [04. Response](#response)
- [05. Summary](#summary)
  
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

# Report Snapshot <a name="snapshot-report"></a>

Following is a sanpshot of the Nike Earnings Call Conference Report. The Report is 29 page long. We summarize this report into 1 page seven bullet points using Open AI and LangChain.

<br>

<image src="/img/posts/Screenshot_Nike.png">

<br>

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
# Load document and initialize model <a name="load-doc-initialize-model"></a>
```python
# Load the document to be summarized
doc_path = "/NIKE-Inc-Q3FY24-OFFICIAL-Transcript-FINAL.pdf"
loader = PyPDFLoader(doc_path)
# PyPDFLoader(filepath).load() reads pdf file, splits text by page, 
# index each page in standardized LangChain document structure 
# with "metadata" and "page_content"
docs= loader.load()


# Read the API KEY from a yml file
OPENAI_KEY = yaml.safe_load(open("../credentials.yml"))['openai']
# Initialize the model
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    api_key=OPENAI_KEY
    )
```

# Prompt Template and LLMChain to run queries against LLM <a name="Prompt-LLMChain"></a>

```python
# EXPANDING WITH PROMPT TEMPLATES
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

# Response <a name="response"></a>
```python
# The chain below takes a list of documents and first combines them into a single string.
# It does this by formatting each document into a string with the `document_prompt` and
# then joining them together with `document_separator`. It then adds that new string to
# the inputs with the variable name set by `document_variable_name`.
# Those inputs are then passed to the `llm_chain`.

chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="text"
        )

response = chain.invoke(docs ,metadata="text")
```

# Summary <a name="summary"></a>

```python
pprint(response['output_text'])
```

'1. NIKE, Inc. reported fiscal 2024 Q3 results in line with expectations, but '
 'acknowledged the need to make adjustments to reach full potential.\n'
 
 '2. The company is focusing on sharpening its focus on sport, driving new '
 'product innovation, enhancing brand marketing, and collaborating with '
 'wholesale partners.\n'
 
 '3. NIKE is building a multiyear cycle of innovation, particularly in the Air '
 'platform, to bring freshness and newness to consumers.\n'
 
 '4. The company is managing its product portfolio through a period of '
 'transition, which may result in near-term headwinds but is expected to drive '
 'long-term growth.\n'
 
 '5. NIKE is accelerating innovation, storytelling, and consumer experience to '
 'drive brand distinction and growth.\n'
 
 '6. The company expects revenue to grow approximately 1% for the full year, '
 'with a focus on expanding gross margins and disciplined cost controls.\n'
 
 '7. NIKE is leveraging the upcoming Olympics as a catalyst for showcasing new '
 'products and driving brand impact and consumer connection through sport.'













