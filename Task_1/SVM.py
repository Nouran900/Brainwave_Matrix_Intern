# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# 1. Load the data
fake = pd.read_csv("Fake.csv", on_bad_lines='skip', quotechar='"', encoding='utf-8')
real = pd.read_csv("True.csv", on_bad_lines='skip', quotechar='"', encoding='utf-8')


# 2. Add labels: 0 for fake, 1 for real
fake['label'] = 0
real['label'] = 1

# 3. Combine and shuffle the data
data = pd.concat([fake, real], ignore_index=True)
data = data[['text', 'label']]  # Use only the text and label columns

# 4. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

# 5. Convert text to numerical format using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Train the Naive Bayes classifier
model = LinearSVC()
model.fit(X_train_vec, y_train)

# 7. Predict and evaluate
y_pred = model.predict(X_test_vec)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the fitted vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("🧠 Model and vectorizer saved as 'model.pkl' and 'vectorizer.pkl'")