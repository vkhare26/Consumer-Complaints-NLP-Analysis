# Consumer-Complaints-NLP-Analysis

# Project Overview

The objective is to analyze consumer financial complaints from the Consumer Financial Protection Bureau (CFPB) dataset using Natural Language Processing (NLP) techniques. The dataset includes unstructured complaint narratives and metadata (e.g., product, issue, company, state) for complaints submitted via the CFPB website. The project involves exploratory data analysis, text preprocessing, topic identification, sentiment analysis, and leveraging GPT-3.5 via OpenAI's API to generate summaries and insights. The goal is to uncover trends in consumer complaints, predict sentiment, and provide actionable recommendations to improve financial services.
# Methodology
The analysis is conducted in a Jupyter Notebook (DSPM_HW3_vinayakk.ipynb) using Python, following the assignment's four main tasks:

## Text Preprocessing and Data Profiling:

Load the CFPB dataset, handling encoding issues (e.g., using encoding='ISO-8859-1' for UTF-8 errors) with pandas.
Profile the dataset to identify missing values in fields like Consumer complaint narrative, Product, or Company.
Clean the Consumer complaint narrative by:
Tokenizing text using nltk or spaCy.
Removing punctuation, stopwords, and frequent phrases.
Applying lemmatization (preferred for contextual accuracy) or stemming to extract word roots.


Identify the most common root words in the cleaned text to understand prevalent complaint themes.


## Topic Identification:

Analyze the Product, Sub-product, and Issue fields to identify the 10 most common complaint topics.
Use pandas to group and count occurrences at each level (e.g., Product: credit reporting, Issue: incorrect information).
Create visualizations (e.g., bar charts or pie charts) with matplotlib or seaborn to display the most frequent categories, highlighting trends in consumer issues.


## Sentiment Analysis:

### Step 3.1: VADER Sentiment Scoring:
Apply the VADER sentiment analyzer (nltk.sentiment.vader) to the cleaned complaint narratives.
Assign sentiment scores on a 1-5 scale based on compound scores: 1 (< -0.5), 2 (-0.5 to -0.1), 3 (-0.1 to 0.1), 4 (0.1 to 0.5), 5 (> 0.5).


### Step 3.2: Sentiment Prediction Model:
Build a supervised learning model (e.g., Logistic Regression, Random Forest) to predict sentiment scores (1-5) using lemmatized/stemmed words as features, represented via TF-IDF or word embeddings.
Identify the top predictive words for each sentiment rating using feature importance or coefficients.
Display sample complaints for each rating and evaluate their reasonableness based on narrative content.




## GPT-3.5 Analysis:

Use OpenAI’s API with the GPT-3.5 model for three tasks:
Summarization: Prompt GPT-3.5 with sample complaint narratives to generate 1-2 sentence summaries and assess their accuracy and conciseness.
Feedback for Low Sentiment: Select narratives with low sentiment scores (1-2) and prompt GPT-3.5 to explain customer dissatisfaction or suggest constructive feedback for resolution.
Predictive Themes: Provide cleaned text for each sentiment rating and ask GPT-3.5 to identify predictive words and themes, comparing results with the model from Step 3.2.


Ensure the OpenAI API key is securely handled and removed from the notebook before submission.



This methodology combines traditional NLP techniques (tokenization, sentiment analysis, topic modeling) with modern large language model capabilities to extract actionable insights from consumer complaints, enabling better customer service strategies for financial institutions.
