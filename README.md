# Telco Customer Churn Prediction

## Problem Statement
The objective of this project is to predict whether a telecom customer will churn (Yes/No) using customer demographic and service usage data.

This is a binary classification problem.

---

## Dataset

Source: Kaggle – Telco Customer Churn  
Total samples: 7043  
Features: 20+ customer service and demographic attributes  

Target variable:
Churn (Yes/No)

---

## Data Preprocessing

- Removed `customerID`
- Converted `TotalCharges` to numeric
- Handled missing values
- One-hot encoded categorical variables
- Standardized numerical features
- Performed stratified train-validation split (80-20)

---

## Train/Validation Split Method

Used `train_test_split` with:
- test_size = 0.2
- stratify = target variable
- random_state = 42

This ensures class distribution consistency.

---

## Models Used

### 1️⃣ Baseline Model
Logistic Regression

### 2️⃣ Improved Model
Random Forest Classifier

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

---

## Best Results

| Model              | Accuracy | F1 Score | ROC-AUC |
|--------------------|----------|----------|---------|
| Logistic Regression| ~78%     | ~0.72    | ~0.82   |
| Random Forest      | ~83%     | ~0.79    | ~0.88   |

Random Forest performed better due to its ability to capture nonlinear relationships.

---

## Error Analysis

- Most false negatives occur in customers with month-to-month contracts.
- Customers with higher monthly charges and low tenure are more likely to churn.
- Feature importance shows Contract type, Tenure, and MonthlyCharges are strong predictors.

---

## How to Run

Install dependencies:

pip install -r requirements.txt

Train model:

python train.py

Evaluate model:

python evaluate.p
