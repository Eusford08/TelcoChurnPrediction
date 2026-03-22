import streamlit as st
import pandas as pd
import joblib

from src.features import add_features

# Load trained pipeline model
model = joblib.load("model/churn_model.pkl")

st.set_page_config(page_title="Telco Churn Prediction", layout="wide")

st.title("📊 Telco Customer Churn Prediction")
st.markdown("Predict whether a customer is likely to churn.")

# ========================
# INPUT SECTION
# ========================

st.header("🧾 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])

with col2:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

with col3:
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

col4, col5, col6 = st.columns(3)

with col4:
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

with col5:
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

with col6:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

col7, col8 = st.columns(2)

with col7:
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)

with col8:
    total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

# ========================
# PREDICTION
# ========================

if st.button("🔍 Predict Churn"):

    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    input_data = add_features(input_data)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # ========================
    # OUTPUT SECTION
    # ========================

    st.header("📈 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to CHURN\n\nProbability: {probability:.2%}")
    else:
        st.success(f"✅ Customer is NOT likely to churn\n\nProbability: {probability:.2%}")

    # Progress bar
    st.progress(float(probability))

    # Interpretation
    st.subheader("💡 Interpretation")

    if probability > 0.7:
        st.write("High risk customer. Immediate retention action recommended.")
    elif probability > 0.4:
        st.write("Moderate risk. Consider engagement strategies.")
    else:
        st.write("Low risk. Customer likely to stay.")