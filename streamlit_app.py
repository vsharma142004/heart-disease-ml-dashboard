import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
import os
import plotly.graph_objects as go
import pandas as pd

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = os.path.dirname(__file__)

# -------------------------------
# Load model and scaler
# -------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(os.path.join(BASE_DIR, "heart_model.h5"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    return model, scaler

model, scaler = load_model()

# -------------------------------
# Load dataset
# -------------------------------
@st.cache_data
def load_data():

    columns = [
        "age","sex","cp","trestbps","chol","fbs","restecg",
        "thalach","exang","oldpeak","slope","ca","thal","target"
    ]

    df = pd.read_csv(
        os.path.join(BASE_DIR, "processed.cleveland.data"),
        names=columns,
        na_values="?"
    )

    df = df.dropna()

    return df

df = load_data()

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Heart Disease ML Dashboard",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease ML Dashboard")

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3 = st.tabs([
    "❤️ Prediction Dashboard",
    "📊 Model Performance",
    "📂 Dataset Explorer"
])

# =====================================================
# TAB 1 — PREDICTION DASHBOARD
# =====================================================
with tab1:

    st.header("Patient Risk Prediction")

    st.sidebar.header("Patient Information")

    age = st.sidebar.number_input("Age", 1, 120, 40)

    sex = st.sidebar.selectbox(
        "Sex",
        ["Male", "Female"]
    )

    cp = st.sidebar.selectbox("Chest Pain Type", [0,1,2,3])
    trestbps = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.sidebar.number_input("Cholesterol", 100, 400, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar >120", [0,1])
    restecg = st.sidebar.selectbox("Rest ECG", [0,1,2])
    thalach = st.sidebar.number_input("Max Heart Rate", 70, 210, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", [0,1])
    oldpeak = st.sidebar.number_input("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("Slope", [0,1,2])
    ca = st.sidebar.selectbox("Number of Major Vessels", [0,1,2,3])
    thal = st.sidebar.selectbox("Thal", [0,1,2,3])

    sex = 1 if sex == "Male" else 0

    if st.button("Predict Heart Disease Risk"):

        features = np.array([[age, sex, cp, trestbps, chol, fbs,
                              restecg, thalach, exang, oldpeak,
                              slope, ca, thal]])

        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)

        risk = float(prediction[0][0])
        risk_percent = risk * 100

        col1, col2 = st.columns(2)

        with col1:

            if risk > 0.5:
                st.error("⚠️ High Risk of Heart Disease")
            else:
                st.success("✅ Low Risk of Heart Disease")

            st.metric("Risk Percentage", f"{risk_percent:.2f}%")

        with col2:

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_percent,
                title={'text': "Heart Disease Risk"},
                gauge={
                    'axis': {'range': [0,100]},
                    'bar': {'color': "white"},
                    'steps': [
                        {'range': [0,30], 'color': "green"},
                        {'range': [30,60], 'color': "yellow"},
                        {'range': [60,100], 'color': "red"}
                    ],
                }
            ))

            st.plotly_chart(fig, use_container_width=True)

        st.info(
            "⚠️ This prediction is for educational purposes only and not medical advice."
        )

# =====================================================
# TAB 2 — MODEL PERFORMANCE
# =====================================================
with tab2:

    st.header("Model Performance")

    st.subheader("Model Accuracy")

    st.metric("Accuracy", "87%")

    st.write(
        "Accuracy is calculated on the test dataset during model training."
    )

    st.subheader("Target Distribution")

    st.bar_chart(df["target"].value_counts())

# =====================================================
# TAB 3 — DATASET EXPLORER
# =====================================================
with tab3:

    st.header("Dataset Explorer")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.subheader("Dataset Statistics")

    st.write(df.describe())

    st.subheader("Feature Correlation Heatmap")

    corr = df.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu"
        )
    )

    st.plotly_chart(fig, use_container_width=True)