# 📂 Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay
)

# 📥 Load dataset
df_train = pd.read_csv("AIML Dataset.csv")
df_test = pd.read_csv("AIML Dataset.csv")

df = pd.concat([df_train, df_test], ignore_index=True)

print("Dataset shape:", df.shape)
print(df['isFraud'].value_counts())

# 🎯 Target and features
y = df['isFraud']

# Drop columns not useful or ID-like
X = df.drop(columns=['isFraud', 'nameOrig', 'nameDest'])

# One-hot encode categorical features (type is categorical)
X = pd.get_dummies(X, columns=['type'], drop_first=True)

# Save the list of columns for prediction time
joblib.dump(X.columns.tolist(), "trained_columns.pkl")

# 🔷 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🔷 Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler for prediction time
joblib.dump(scaler, "scaler.pkl")

# 🔷 Logistic Regression with class_weight balanced
clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
clf.fit(X_train_scaled, y_train)

# 🔷 Predictions
y_pred = clf.predict(X_test_scaled)
y_proba = clf.predict_proba(X_test_scaled)[:, 1]

# 📊 Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 🔷 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


joblib.dump(clf, "fraud_model.pkl")
print("✅ Model, scaler, and columns saved.")
