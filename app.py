import streamlit as st
import numpy as np
import pandas as pd
import joblib
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# -------------------------- PAGE CONFIG --------------------------
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------- HEADER -------------------------------
st.title("🩺 Diabetes Prediction System")
st.markdown("""
### Welcome to the AI-Powered Diabetes Prediction App  
Enter your medical data below to predict your **risk of diabetes**.  
This app uses a Logistic Regression model trained on the **PIMA Indian Diabetes Dataset**.
""")

# -------------------------- SIDEBAR INPUT -------------------------
st.sidebar.header("📋 Patient Information")

Pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 2)
Glucose = st.sidebar.slider("Glucose Level", 0, 200, 120)
BloodPressure = st.sidebar.slider("Blood Pressure (mm Hg)", 0, 140, 70)
SkinThickness = st.sidebar.slider("Skin Thickness (mm)", 0, 100, 20)
Insulin = st.sidebar.slider("Insulin Level (mu U/ml)", 0, 900, 85)
BMI = st.sidebar.slider("BMI (Body Mass Index)", 0.0, 70.0, 25.0)
DiabetesPedigreeFunction = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
Age = st.sidebar.slider("Age", 1, 120, 30)

# Prepare data
input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                        Insulin, BMI, DiabetesPedigreeFunction, Age]])

scaled_data = scaler.transform(input_data)

# -------------------------- PREDICTION ----------------------------
if st.sidebar.button("🔍 Predict Diabetes Risk"):
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0][1]
    
    # Risk label
    if probability < 0.4:
        risk = "Low Risk"
        color = "green"
    elif probability < 0.7:
        risk = "Moderate Risk"
        color = "orange"
    else:
        risk = "High Risk"
        color = "red"

    # ------------------ DISPLAY RESULTS --------------------------
    st.markdown("## 🧾 Prediction Result")
    st.markdown(f"### **Prediction:** :{color}[{risk}]")
    st.progress(float(probability))
    st.metric(label="Diabetes Probability", value=f"{probability:.2f}")

    # ------------------ VISUALIZATION 1: Probability Bar -----------------
    st.markdown("### 📊 Diabetes Probability Visualization")
    prob_df = pd.DataFrame({
        'Risk': ['No Diabetes', 'Diabetes'],
        'Probability': [1 - probability, probability]
    })
    chart = alt.Chart(prob_df).mark_bar(width=50).encode(
        x='Risk',
        y='Probability',
        color=alt.Color('Risk', scale=alt.Scale(domain=['No Diabetes', 'Diabetes'], range=['#4CAF50', '#E74C3C']))
    ).properties(height=300)
    st.altair_chart(chart, use_container_width=True)

    # ------------------ VISUALIZATION 2: Feature Radar -----------------
    st.markdown("### 🕸️ Patient Profile Overview")
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    values = input_data.flatten().tolist()

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    ax.plot(angles, values, color='red', linewidth=2)
    ax.fill(angles, values, color='red', alpha=0.25)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=9)
    plt.title("Patient Health Metrics", y=1.1)
    st.pyplot(fig)

# -------------------------- EXPLANATION SECTION --------------------------
st.markdown("---")
st.markdown("## 🧠 Model Insights")

# Coefficients as feature importance
coefficients = pd.DataFrame({
    'Feature': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
    'Importance': model.coef_[0]
}).sort_values(by='Importance', ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=coefficients, palette='coolwarm', ax=ax)
plt.title("Feature Importance (Logistic Regression Coefficients)")
st.pyplot(fig)

st.info("""
**Interpretation:**  
- Positive values → increase diabetes risk  
- Negative values → decrease risk  
""")

# -------------------------- FOOTER --------------------------
st.markdown("""
---
👨‍⚕️ **Developer:** Hemasai Reddy  
💡 *Powered by Logistic Regression + Streamlit + Machine Learning*
""")
