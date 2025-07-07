import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and trained columns
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
trained_columns = joblib.load("trained_columns.pkl")

# 📊 Title
st.title("💳 Fraud Detection")

st.markdown("Please enter the transaction details and click **Predict** to see if it is likely fraud.")

st.divider()

# Input fields
transaction_type = st.selectbox("Transaction Type", [
    "PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"
])
amount = st.number_input("Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, value=10000.0)
newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0, value=9000.0)
oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0, value=0.0)

if st.button("Predict"):
    # Create input dataframe
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    # One-hot encode 'type'
    input_data = pd.get_dummies(input_data)

    # Align columns
    input_data = input_data.reindex(columns=trained_columns, fill_value=0)

    # Scale numeric values
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    st.subheader(f"Prediction: {int(prediction)}")

    if prediction == 1:
        st.error("🚨 This transaction is predicted to be **fraudulent!**")
    else:
        st.success("✅ This transaction is predicted to be **legitimate.**")
