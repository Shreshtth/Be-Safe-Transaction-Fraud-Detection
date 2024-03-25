The fraudulent transaction detection project involves creating a synthetic dataset based on transaction data and designing Machine/Deep Learning models to flag fraudulent transactions. The models used include XGBoost, Random Forest, and DNN-Keras, with a focus on achieving high accuracy and precision/recall for fraud detection.

### Sample Transaction:
- **Transaction ID:** 532684489994665409
- **Transaction Date:** 2024-03-06 20:28:58:000
- **Transaction Type:** UPI
- **Transaction Amount:** $1
- **Payer VPA:** saikrishnareddy703231@ybl
- **Payee VPA:** 123@bank
- **Status:** SUCCESS
- **Customer Name:** JANGA SAI KRISHNA

---

# Fraud Detection Model using Machine Learning

## Overview
This project focuses on creating a synthetic dataset based on transaction data and designing Machine/Deep Learning models to detect fraudulent transactions. The models used include XGBoost, Random Forest, and DNN-Keras, with the aim of achieving high accuracy and precision/recall for fraud detection.

## Dataset Generation and Data Preprocessing
- Features such as transaction amount, transaction type, status, payer_vpa, payee_vpa, ip_address, hour, and date were used in dataset generation.
- Feature engineering involved creating a new feature called "Transaction Frequency" based on existing parameters.
- Data preprocessing steps included converting transactions to a DataFrame, encoding transaction types, and saving the dataset to a CSV file.

## Model Assignment
- **XGBoost:** A powerful algorithm that corrects errors made by previous models to improve prediction accuracy.
- **Random Forest:** An ensemble learning method that builds multiple decision trees for more reliable predictions.
- **DNN-Keras:** A type of artificial neural network that learns intricate patterns from complex data.

## Model Evaluation
- **XGBoost:** Accuracy - 0.76, ROC AUC Score - 0.64
- **Random Forest:** F1 Score - 0.86, ROC AUC Score - 0.64
- **DNN-Keras:** Accuracy - 0.77, ROC AUC Score - 0.63

## Suggestions for Improvement
- Use a more balanced and real-life dataset for evaluation.
- Experiment with different hyperparameters of the XGBoost model.
- Apply balancing techniques like undersampling and oversampling.
- Explore additional feature engineering possibilities to enhance fraud detection.

---

### Citations:
1. https://github.com/sharmasapna/credit-card-fraud-detection : Dataset created using Sparkov Data Generation tool created by Brandon Harris. Multiple models applied- autoencoders, DNN, TSNE, isolation forest,
2. https://github.com/lmriccardo/fraudolent-transaction-classification : Project for the Big Data Computing course at the University of "La Sapienza" in master’s in computer science A.A. 2021/2022. Models used are Decision Trees, Random Forest, Logistic Regression, Gradient Boosted Tree, and Neural Network approach.
3. https://www.kaggle.com/c/ieee-fraud-detection/data : Dataset used - Vesta's real-world e-commerce transactions and contains a wide range of features from device type to product features.
4. https://ieeexplore.ieee.org/abstract/document/9178971 : AdaBoost is a machine learning method that uses boosting. It assumes that the data in training and testing sets have the same distribution and input feature space. It increases the weights of instances that are wrongly classified in the training process. AdaBoost is extended to transfer AdaBoost (TrAdaBoost) that can effectively transfer knowledge from one domain to another.

---
