# IBM Telco Customer Churn Prediction

Predict customer churn using Machine Learning classification algorithms with complete data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and comparison.

---

## Project Overview

Customer churn is one of the biggest challenges faced by telecom companies. This project analyzes customer behavior and builds Machine Learning models to predict whether a customer is likely to leave the telecom service.

The notebook follows a complete end-to-end Machine Learning pipeline from data preprocessing to model evaluation and comparison.

---

## Problem Statement

Develop a classification model that can accurately predict whether a customer will churn based on customer demographics, account information, and subscribed services.

---

## Dataset

**Dataset Name:** IBM Telco Customer Churn Dataset

**Target Variable:**
- Churn

### Features Include

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Online Security
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

---

## Project Workflow

### 1. Import Libraries

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Joblib

---

### 2. Data Loading

- Load Dataset
- Display Dataset
- Check Shape
- Dataset Information
- Statistical Summary

---

### 3. Exploratory Data Analysis (EDA)

- Missing Value Analysis
- Duplicate Check
- Target Distribution
- Univariate Analysis
- Bivariate Analysis
- Correlation Analysis
- Business Insights

---

### 4. Data Preprocessing

- Handle Missing Values
- Convert Data Types
- Encode Categorical Variables
- Feature Scaling
- Train-Test Split

---

### 5. Handling Class Imbalance

- Apply SMOTE
- Balance Minority Class

---

### 6. Machine Learning Models

The following classification algorithms were implemented:

- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Support Vector Machine (SVM)
- Gaussian Naive Bayes

---

### 7. Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- Classification Report
- Cross Validation

---

### 8. Model Comparison

Different models were compared to identify the best-performing classifier for churn prediction.

---

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Joblib

---

## Machine Learning Pipeline

```
Dataset
    │
    ▼
Data Cleaning
    │
    ▼
EDA
    │
    ▼
Feature Encoding
    │
    ▼
Feature Scaling
    │
    ▼
Train-Test Split
    │
    ▼
SMOTE
    │
    ▼
Model Training
    │
    ▼
Model Evaluation
    │
    ▼
Model Comparison
    │
    ▼
Best Model
```

---

## Business Benefits

- Predict customers likely to churn.
- Improve customer retention.
- Reduce revenue loss.
- Enable targeted marketing campaigns.
- Support business decision-making.

---

## Requirements

Install required libraries using:

```bash
pip install -r requirements.txt
```

---

## Results

The project compares multiple Machine Learning classification algorithms and selects the best-performing model based on evaluation metrics. The final model can be used to predict customer churn and support customer retention strategies.

---
