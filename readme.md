# Reddit Sentiment Analysis

## Overview

This project focuses on sentiment classification of Reddit text posts. The objective is to categorize user-generated content into sentiment classes by leveraging transformer-based language models. The primary model used in this work is **bert-base-uncased**, fine-tuned specifically on sentiment-labeled Reddit and Twitter data.

## Dataset

Dataset used:
[https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset)
The dataset contains labeled text samples spanning multiple sentiment classes, providing a balanced foundation for model fine-tuning.

## Models and Performance

* **Fine-tuned Model:** bert-base-uncased
* **Evaluation Result:** F1 score ≈ **93%**
* **Naive Bayes Baseline:** Accuracy ≈ **62%**

The fine-tuned BERT model demonstrated a significant performance improvement over classical baseline methods, indicating the value of contextual language representations for sentiment tasks.

## Model Access

The fine-tuned model weights are publicly available:
[https://huggingface.co/AtulDeshpande/reddit_sentiment_model/tree/main](https://huggingface.co/AtulDeshpande/reddit_sentiment_model/tree/main)

## Demo

An interactive inference Space is provided for testing the model on custom input text:
[https://huggingface.co/spaces/AtulDeshpande/reddit_sentiment](https://huggingface.co/spaces/AtulDeshpande/reddit_sentiment)
