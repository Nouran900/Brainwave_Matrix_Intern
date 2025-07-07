# 📂 Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

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

# 🔷 Downsample majority class for speed
rus = RandomUnderSampler(sampling_strategy=0.1, random_state=42)  # keep 10% of majority class
X_train_small, y_train_small = rus.fit_resample(X_train_scaled, y_train)

print(f"After undersampling: {np.bincount(y_train_small)}")

# 🔷 Apply SMOTE to create a balanced, small training set
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_small, y_train_small)

print(f"After SMOTE: {np.bincount(y_train_res)}")

# 🔷 Faster Random Forest
clf = RandomForestClassifier(
    n_estimators=30,         # fewer trees
    max_depth=10,            # shallower trees
    class_weight='balanced',
    random_state=42,
    n_jobs=-1                # use all CPU cores
)
clf.fit(X_train_res, y_train_res)

# 🔷 Predictions with threshold tuning
y_proba = clf.predict_proba(X_test_scaled)[:, 1]

# Plot Precision-Recall curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
plt.plot(thresholds, precisions[:-1], label='Precision')
plt.plot(thresholds, recalls[:-1], label='Recall')
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision vs Recall")
plt.legend()
plt.show()

# Choose a threshold
threshold = 0.7
y_pred = (y_proba >= threshold).astype(int)

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
print("✅ Faster Random Forest model, scaler, and columns saved.")
