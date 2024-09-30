# Sentiment Analysis System for Hospital - Deep Learning Project

## Overview
This project is part of the Final Assessment for Deep Learning course. The task focuses on building a Sentiment Analysis System for a hospital in Indonesia. The system aims to extract emotions from patients' questionnaire responses using a dataset of textual inputs and sentiment labels (`Emotion.csv`).

The goal of the project is to preprocess the data, build a classification model, and evaluate its performance in terms of sentiment prediction accuracy.

## Dataset
- Filename: `Emotion.csv`
- Columns of Interest:
  - `Text`: Textual responses from patients.
  - `Sentiment`: The corresponding sentiment label for each response, which can represent emotions such as "positive", "negative", "neutral", etc.
 
## Key Tasks
1. **Data Preprocessing and Preparation**
The first step in building a robust sentiment analysis system is to preprocess the raw text data. Several NLP (Natural Language Processing) techniques are employed to prepare the data:
    - **Tokenization**: Breaking down each sentence into individual words (tokens).
    - **Lowercasing**: Converting all text to lowercase for uniformity.
    Stopwords Removal: Removing common but irrelevant words (e.g., "and", "the") that do not contribute to sentiment classification.
    - **Lemmatization/Stemming**: Reducing words to their base or root form to group similar words together (e.g., "running" becomes "run").
    - **Vectorization**: Converting text into numerical format using methods like TF-IDF (Term Frequency-Inverse Document Frequency) or word embeddings (e.g., Word2Vec, GloVe) for input to machine learning models.

2. **Dataset Partitioning**
The dataset is split into three parts to ensure the model generalizes well to unseen data:
    - Training Set: 70% of the data is used to train the model.
    - Validation Set: 15% of the data is used to tune hyperparameters and evaluate the model's performance during training.
    - Test Set: 15% of the data is reserved for evaluating the final model's performance.

3. **Model Development**
The model BERT (provided in the project notebook) and is designed to classify the text into corresponding sentiment categories. The architecture may include layers such as:
    - Embedding Layer: Converts each word into dense vectors.
    - Recurrent Layers (LSTM/GRU): Captures the sequential nature of the text data and helps model long-term dependencies in sentiment.
    - Fully Connected Layers: Maps learned features to the final sentiment output.

4. **Model Evaluation and Performance Metrics**
The performance of the model is assessed on the test set using the following evaluation metrics:
    - Accuracy: The proportion of correct predictions out of all predictions made by the model.
    - Precision: The proportion of true positive predictions (correctly predicted sentiment) out of all positive predictions.
    - Recall: The proportion of actual positive examples that were correctly predicted by the model.
    - F1-Score: The harmonic mean of precision and recall, used as a balanced measure of model performance

## Conclusion
This project demonstrates how sentiment analysis can be applied to real-world hospital questionnaire data. By preprocessing text data and building a deep learning model, the project provides valuable insights into how emotions can be extracted from written responses.
