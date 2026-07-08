# Telco Customer Churn - Supervised Learning Project

## Project Overview

This project is a Practical Exam submission for the "Customer Churn
Prediction - Multi-Algorithm Classification Showdown" assignment. Acting as
a Junior Data Scientist at a telecom company, the goal is to build and
compare four supervised learning classification models to predict which
customers are likely to churn (leave the service), and to recommend which
model the retention team should deploy.

The project covers the full machine learning lifecycle: exploratory data
analysis, feature engineering, handling class imbalance, training and
tuning four different classifiers (KNN, Naive Bayes, SVM, Decision Tree),
model evaluation and comparison, error analysis, and packaging the final
model into a reusable, deployable pipeline.

## Dataset Information

- **Dataset:** IBM Telco Customer Churn Dataset
- **Source:** [Kaggle - blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Rows:** 7,043 customers
- **Columns:** 21 (demographics, subscribed services, contract, billing)
- **Target:** `Churn` (Yes = churned, No = retained) - binary classification

## Features

Key raw features include customer demographics (`gender`, `SeniorCitizen`,
`Partner`, `Dependents`), account information (`tenure`, `Contract`,
`PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`), and
subscribed services (`PhoneService`, `MultipleLines`, `InternetService`,
`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`,
`StreamingTV`, `StreamingMovies`).

Engineered features created during preprocessing:

- `tenure_group` - New (0-12 months), Mid (13-36), Senior (37-60), Loyal (61+)
- `num_services` - count of add-on services the customer subscribes to
- `AutoPay` - flag for automatic payment methods

## Workflow

1. **Problem Framing & Theory Notes** - business context and key ML concepts.
2. **Dataset Loading & EDA** - univariate and bivariate analysis, correlation
   heatmap, class balance check.
3. **Data Preprocessing & Feature Engineering** - cleaning, feature creation,
   encoding, scaling, SMOTE, train-test split.
4. **Model Building - KNN & Naive Bayes** - baseline, hyperparameter tuning,
   evaluation.
5. **Model Building - SVM & Decision Tree** - baseline, hyperparameter
   tuning, evaluation, SMOTE vs class_weight comparison.
6. **Model Evaluation & Comparison** - consolidated metrics table, combined
   ROC curves, Precision vs Recall chart, final recommendation.
7. **Error Analysis & Interpretation** - False Negative profiling, feature
   importance interpretation.
8. **Pipeline, Deployment & GitHub Submission** - final sklearn Pipeline,
   saved with Joblib, reloaded and tested on sample customers.

## Models Used

- K-Nearest Neighbours (KNN)
- Gaussian Naive Bayes
- Support Vector Machine (SVM, RBF kernel)
- Decision Tree Classifier

All four models were tuned using cross-validation and evaluated on the same
held-out, untouched test set for a fair comparison.

## Results

A consolidated comparison table (Accuracy, Precision, Recall, F1-Score,
AUC-ROC, Training Time) for all four models is available in Section 6 of
the notebook, sorted by Recall on the Churn class (the top business
priority, since missing a real churner is costlier than a false alarm). The
full analysis, charts, and final recommendation are documented in the
notebook and in `summary_report.md`.

## Folder Structure

```
telco-churn-supervised-learning/
|-- CustomerChurn_SupervisedLearning.ipynb   Fully executed notebook
|-- churn_model.pkl                          Saved sklearn Pipeline (best model)
|-- summary_report.md                        ~400-500 word summary report
|-- requirements.txt                         Python dependencies
|-- README.md                                This file
|-- data/
|   `-- WA_Fn-UseC_-Telco-Customer-Churn.csv Raw dataset
`-- charts/
    `-- *.png                                All saved EDA and evaluation charts
```

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/<your-username>/telco-churn-supervised-learning.git
   cd telco-churn-supervised-learning
   ```
2. (Recommended) Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate      # on Windows: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## How to Run

1. Make sure `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` is present (already
   included in this repository).
2. Launch Jupyter Notebook:
   ```
   jupyter notebook
   ```
3. Open `CustomerChurn_SupervisedLearning.ipynb` and run all cells from top
   to bottom (Kernel > Restart & Run All). This will regenerate every chart
   and re-train/re-save `churn_model.pkl`.
4. To reuse the saved model directly without re-running the notebook:
   ```python
   import joblib
   pipeline = joblib.load("churn_model.pkl")
   predictions = pipeline.predict(new_customer_dataframe)
   probabilities = pipeline.predict_proba(new_customer_dataframe)[:, 1]
   ```

## Requirements

See `requirements.txt` for the full list of Python package dependencies
(pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn,
joblib, jupyter).

## Video Walkthrough

Video link: _add your recorded video URL here before submission_

---
*Practical Exam - Supervised Learning (Set B) - Red & White Skill Education*
