# app.py
import streamlit as st
import pickle

# Load saved model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit UI
st.title("📰 Fake News Detection App")
st.write("Paste a news article below and check if it's real or fake:")

text_input = st.text_area("Enter article text:")

if st.button("Check"):
    if text_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        text_vector = vectorizer.transform([text_input])
        prediction = model.predict(text_vector)[0]
        label = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
        st.success(f"Prediction: {label}")
