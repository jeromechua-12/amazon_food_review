# Sentimental Analysis with RoBERTa
![build](https://img.shields.io/badge/build-passing-brightgreen)
![project-type](https://img.shields.io/badge/project_type-NLP-orange)
![python](https://img.shields.io/badge/python-3.12.10-blue)

This project performs **sentiment analysis** on the
[Amazon Fine Food Reviews dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
using a fine-tuned **RoBERTa** model from Hugging Face's Transformers library.

## Project Purpose
The goal of this project is to classify customer reviews into three sentiment categories:
- **Negative** (Scores 1-2)
- **Neutral** (Score 3)
- **Positive** (Scores 4-5)

## Workflow
1. Labelling scores
1. Splitting data into training, validation and test data.
   - Stratified split was done to ensure proportion of classes are maintained.
1. Tokenisation of reviews (text data) using RoBERTa pre-trained tokenizer.
1. Batch processing of input data as Hugging Face's Datasets.
1. Load a pre-trained RoBERTa model.
1. Fine-tune the model using Hugging Face's Trainer
    - Trainer handles necessary components of a typical training loop.
      - Loss function
      - Gradient calculation 
      - Update weights and biases based on gradient using an optimiser.
    - Cross-entropy loss is used as the loss function while AdamW is the optimiser.
1. Evaluate metrics containing accuracy, precision, recall and f1 score.

## Tools & Libraries
- NumPy - For numerical operations
- Pandas - For data processing and exploration
- Hugging Face Transformers - For tokenization and model fine-tuning (RoBERTa)
- Hugging Face Datasets - For optimsied data preprocessing
- Hugging Face Trainer - Simplified and optimised training pipeline that works with Hugging Face Transformers.
- Scikit-learn (sklearn) - For splitting of data and evaluation metrics (accuracy, precision, recall, and F1-score)

## Model Performance
The trained model achieved an overall **accuracy of 86%** on the test data.


### Improvements
- Due to imbalanced data, the "Neutral" class was not learnt by the model. Can be improved by:
  - Applying Synthetic Minority Oversampling Technique (SMOTE)
  - Perform more epoch
