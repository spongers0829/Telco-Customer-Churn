import pandas as pd
import joblib

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load models
rf_model = joblib.load("models/random_forest_model.pkl")
log_model = joblib.load("models/logistic_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Logistic predictions
X_test_scaled = scaler.transform(X_test)
log_preds = log_model.predict(X_test_scaled)

# Random Forest predictions
rf_preds = rf_model.predict(X_test)

print("===== Logistic Regression =====")
print(confusion_matrix(y_test, log_preds))
print(classification_report(y_test, log_preds))
print("ROC-AUC:", roc_auc_score(y_test, log_model.predict_proba(X_test_scaled)[:,1]))

print("\n===== Random Forest =====")
print(confusion_matrix(y_test, rf_preds))
print(classification_report(y_test, rf_preds))
print("ROC-AUC:", roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1]))
