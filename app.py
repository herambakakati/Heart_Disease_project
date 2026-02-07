import streamlit as st
import joblib
import numpy as np
import os

if not os.path.exists("heart_disease_model.pkl"):
    st.error("❌ Model file not found in repository")
    st.stop()

model = joblib.load("heart_disease_model.pkl")

scaler = None
if os.path.exists("scaler.pkl"):
    scaler = joblib.load("scaler.pkl")


# --------------------------------------------------
# Load trained model and scaler
# --------------------------------------------------
heart_model = joblib.load("heart_disease_model.pkl")
heart_scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Failure Prediction", layout="centered")

st.title("Heart Failure Prediction using ML")

st.image("https://th.bing.com/th/id/R.ce5e9d964f027f5859ca8613f33b2e1c?rik=%2bI%2bJM95axu0bZg&riu=http%3a%2f%2fmedia.clinicaladvisor.com%2fimages%2f2017%2f03%2f29%2fheartillustrationts51811362_1191108.jpg&ehk=5j%2bzmEbCqhDShHbQaCiXdMZby7zNV7EghLdnSeu%2fGNM%3d&risl=&pid=ImgRaw&r=0",
    caption="Human Heart Anatomy",
    width=250
)

st.markdown("Enter patient clinical details below:")

# --------------------------------------------------
# Input fields (MATCH DATASET ORDER)
# --------------------------------------------------
age = st.slider("Age", 1, 120, 60)
anaemia_label = st.selectbox("Anaemia", ["No", "Yes"])
anaemia = 0 if anaemia_label == "No" else 1
creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase", 0, 8000)
diabetes = st.radio("Diabetes", [0, 1], format_func=lambda x: "Yes" if x else "No")
ejection_fraction = st.slider("Ejection Fraction", 0, 100, 38)
high_blood_pressure = st.radio("High Blood Pressure", [0, 1], format_func=lambda x: "Yes" if x else "No")
platelets = st.number_input("Platelets", 0.0, 1000000.0, 263000.0)
serum_creatinine = st.number_input("Serum Creatinine", 0.0, 10.0, 1.2)
serum_sodium = st.number_input("Serum Sodium", 0, 200, 137)
sex_label = st.selectbox("Sex", ["Female", "Male"])
sex = 0 if sex_label == "Female" else 1
smoking_label = st.selectbox("Smoking", ["No", "Yes"])
smoking = 0 if smoking_label == "No" else 1
time = st.number_input("Follow-up Time (days)", 0, 300, 120)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Predict"):
    input_data = np.array([[  
        age,
        anaemia,
        creatinine_phosphokinase,
        diabetes,
        ejection_fraction,
        high_blood_pressure,
        platelets,
        serum_creatinine,
        serum_sodium,
        sex,
        smoking,
        time
    ]])

    # Apply scaling (must match training)
    input_scaled = heart_scaler.transform(input_data)

    prediction = heart_model.predict(input_scaled)[0]
    probability = heart_model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Death Event\n\nProbability: {probability:.2%}")
    else:
        st.success(f"✅ Low Risk of Death Event\n\nProbability: {probability:.2%}")

