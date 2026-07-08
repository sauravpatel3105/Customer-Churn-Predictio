# Summary Report - Customer Churn Prediction

## Business Problem and Dataset Summary

Telecom companies such as Jio or Airtel lose a meaningful share of their
subscriber base every month, and it is far cheaper to retain an existing
customer than to acquire a new one to replace them. This project uses the
IBM Telco Customer Churn dataset (7,043 customers, 21 features covering
demographics, subscribed services, contract and billing details) to build a
supervised learning system that predicts which customers are likely to
churn. About 26.5% of customers in the dataset have churned, making this a
moderately imbalanced binary classification problem. The goal is to give
the retention team a ranked list of at-risk customers so they can act before
those customers leave.

## Preprocessing and Class Imbalance Strategy

The `TotalCharges` column was converted from text to numeric, and the small
number of missing values (all belonging to brand-new customers with zero
tenure) were filled with the median. Three new features were engineered:
`tenure_group` (New/Mid/Senior/Loyal buckets), `num_services` (count of
add-on services subscribed to), and `AutoPay` (automatic payment flag).
Binary Yes/No columns were mapped to 1/0, nominal categorical columns
(`InternetService`, `Contract`, `PaymentMethod`) were one-hot encoded, and
`tenure_group` was label encoded to preserve its natural order. All
numerical features were standardized with `StandardScaler` for a fair
comparison across models. Data was split 80/20 with stratification, and
SMOTE was applied strictly to the training set only, so the test set stayed
fully unaugmented and representative of real-world, imbalanced incoming
data.

## Recommended Model and Metric Evidence

Four classifiers were trained and tuned: KNN, Gaussian Naive Bayes, SVM
(RBF kernel), and Decision Tree. Since a missed churner (False Negative) is
far more costly to the business than an unnecessary retention call (False
Positive), Recall on the Churn class was used as the primary ranking
metric. Naive Bayes and Decision Tree produced the highest Recall scores
among the four models, both comfortably ahead of KNN and SVM at the same
decision threshold. The Decision Tree also offers a practical edge in
production: it is easy to explain to non-technical stakeholders through its
visible if-else rules and feature importances, while SVM behaves as a
black box. We recommend deploying the higher-Recall model (Naive Bayes or
Decision Tree, depending on the exact run) and using its predicted
probabilities to rank customers, so the retention team can prioritize the
riskiest customers within their daily call capacity.

## Top 3 Churn Signals and Retention Actions

1. **Month-to-month contracts** show by far the highest churn rate compared
   to One year or Two year contracts. Action: incentivize month-to-month
   customers with a discount to upgrade to a longer commitment.
2. **Low tenure (new customers)** churn far more than long-tenured
   customers. Action: strengthen onboarding and early engagement in the
   first 12 months, the highest-risk window.
3. **High MonthlyCharges with few add-on services** signals customers who
   feel they are paying a lot without matching perceived value. Action:
   proactively offer bundled services or a loyalty discount to customers in
   this segment.

## Next Steps

To take this further, the team could gather more behavioral data (customer
service call logs, network downtime experienced by each customer, app usage
frequency) to build richer features. A deep learning model or gradient
boosted trees (XGBoost/LightGBM) could be tested for a possible performance
gain now that a solid baseline exists. Finally, wrapping the saved pipeline
in a real-time scoring API would let the retention team query churn risk
for any customer on demand, directly from their CRM system, rather than
running batch predictions manually.
