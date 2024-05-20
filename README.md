# Credit Card Fraud Detection Project

## Overview

This project aims to detect fraudulent transactions in credit card data using logistic regression. The dataset used for this project is the Credit Card Fraud Detection dataset, which contains transactions made by credit cards in September 2013 by European cardholders.

## Dataset

The dataset used in this project is obtained from [Kaggle](https://www.kaggle.com/code/prabhaa/credit-card-dataset). The dataset contains the following:

- `credit_card_data.csv`

## Project Structure

1. **Importing Libraries**
2. **Reading Dataset**
3. **Data Preprocessing**
    - Splitting features and target
    - Standardizing features
    - Adding bias term
4. **Train-Test Split**
5. **Logistic Regression Implementation**
    - Sigmoid function
    - Initialization of parameters
    - Training loop
6. **Model Training and Evaluation**
    - Cost computation
    - Gradient computation and parameter update
    - Evaluation metrics: Accuracy, Precision, Recall, F1 Score
7. **Results**

## Results

The logistic regression model was trained on the credit card dataset and evaluated every 100 epochs. The evaluation metrics include accuracy, precision, recall, and F1 score, which provide a comprehensive understanding of the model's performance in detecting fraudulent transactions.

## Conclusion

This project demonstrated the implementation of logistic regression for credit card fraud detection. The dataset from Kaggle was utilized to train and evaluate the model, achieving promising results. Future work could involve exploring more advanced models and techniques to further improve detection accuracy.
