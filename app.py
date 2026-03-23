import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

from src.features import add_features
from src.cleaning import clean_data
# ========================
# CONFIG
# ========================
st.set_page_config(page_title="Telco Churn Prediction", layout="wide")

# ========================
# LOAD MODEL & DATA
# ========================
@st.cache_resource
def load_model():
    return joblib.load("model/churn_model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("data/train.csv")
    return df

model = load_model()
df = load_data()

# ========================
# TITLE
# ========================
st.title("📊 Telco Customer Churn Prediction")
st.markdown("Interactive dashboard + prediction system for telecom churn analysis")

# ========================
# TABS
# ========================
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 Prediction", "📘 About"])

# =========================================================
# 📊 TAB 1 — DASHBOARD
# =========================================================
with tab1:

    st.header("📊 Customer Overview")

    df_dashboard = df.copy()
    df_dashboard = clean_data(df_dashboard)
    df_dashboard = add_features(df_dashboard)

    # KPI METRICS
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", len(df_dashboard))
    col2.metric("Churn Rate", f"{df_dashboard['Churn'].mean()*100:.2f}%")
    col3.metric("Avg Monthly Charges", f"${df_dashboard['MonthlyCharges'].mean():.2f}")

    st.divider()

    # FILTERS
    st.subheader("🔎 Filters")

    col1, col2 = st.columns(2)

    with col1:
        contract_filter = st.selectbox(
            "Contract Type",
            ["All"] + list(df_dashboard["Contract"].unique())
        )

    with col2:
        tenure_range = st.slider("Tenure Range", 0, 72, (0, 72))

    filtered_df = df_dashboard.copy()

    if contract_filter != "All":
        filtered_df = filtered_df[filtered_df["Contract"] == contract_filter]

    filtered_df = filtered_df[
        filtered_df["tenure"].between(tenure_range[0], tenure_range[1])
    ]

    st.divider()

    # VISUALS
    st.subheader("📈 Insights")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.histogram(
            filtered_df,
            x="tenure",
            color="Churn",
            nbins=20,
            title="Tenure Distribution by Churn"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.box(
            filtered_df,
            x="Churn",
            y="MonthlyCharges",
            title="Monthly Charges vs Churn"
        )
        st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# 🤖 TAB 2 — PREDICTION
# =========================================================
with tab2:

    st.header("🤖 Customer Churn Prediction")

    # INPUT SECTION
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
    # PREDICT
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

        st.divider()

        # RESULT
        if prediction == 1:
            st.markdown("## 🔴 High Risk: Customer Likely to Churn")
        else:
            st.markdown("## 🟢 Low Risk: Customer Likely to Stay")

        st.metric("Churn Probability", f"{probability:.2%}")
        st.progress(float(probability))

        # INTERPRETATION
        st.subheader("💡 Interpretation")

        if probability > 0.7:
            st.warning("High risk → Immediate retention action recommended")
        elif probability > 0.4:
            st.info("Moderate risk → Consider engagement strategies")
        else:
            st.success("Low risk → Customer likely to stay")

    # ========================
    # FEATURE IMPORTANCE
    # ========================
    st.divider()
    st.subheader("📊 Feature Importance")

    model_step = model.named_steps["model"]

    if hasattr(model_step, "feature_importances_"):
        importances = model_step.feature_importances_
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()

        feature_importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        })

        feature_importance["Feature"] = feature_importance["Feature"].apply(
            lambda x: x.split("__")[-1]
        )

        feature_importance["feature_group"] = feature_importance["Feature"].apply(
            lambda x: x.split("_")[0]
        )

        grouped_importance = feature_importance.groupby("feature_group")["Importance"].sum().reset_index()
        grouped_importance = grouped_importance.sort_values(by="Importance", ascending=True)

        fig = px.bar(
            grouped_importance.tail(10),
            x="Importance",
            y="feature_group",
            orientation="h"
        )

        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# 📘 TAB 3 — ABOUT
# =========================================================
with tab3:

    st.header("📘 Project Overview")
    col1, col2, spacer = st.columns([2, 1, .5])
    with col1:
        st.subheader("🎯 Objective")
        st.write("Predict customer churn in a telecom company to support retention strategies.")

        st.subheader("📊 Dataset")
        st.write("""
                Telco Customer Churn dataset containing:
                - Customer demographics
                - Account information
                - Service usage
                """)

        st.subheader("🤖 Models Used")
        st.write("""
                - XGBoost (final selected model)
                - Logistic Regression
                - Random Forest
                - Decision Tree
                - K Nearest Neighbors
                - Support Vector Machines
                """)
    with col2:
        st.markdown("### 📈 Evaluation Metrics")
        st.info("ROC-AUC\n\nRecall (important for churn detection)")

        st.markdown("### 💡 Key Insights")
        st.success("""
        - Month-to-month → highest churn  
        - High charges → higher risk  
        - Low tenure → high churn  
        """)

        st.markdown("### ⚠️ Limitations")
        st.warning("""
        - Imbalance handled with SMOTE  
        - Limited generalization  
        """)

    # st.subheader("🚀 How to Use")
    # st.markdown("""
    # 1. Explore data in Dashboard tab
    # 2. Predict churn in Prediction tab
    # """)