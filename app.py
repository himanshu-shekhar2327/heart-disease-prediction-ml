import streamlit as st
import pandas as pd
import joblib


# -----------------------------
# Load the trained model
# -----------------------------
model = joblib.load("heart_model.pkl")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to get a prediction using the trained ML model.")

# -----------------------------
# User Input Fields
# -----------------------------

age = st.number_input("Age", min_value=1, max_value=120, value=40)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])

resting_bp = st.number_input("Resting BP", min_value=60, max_value=200, value=120)
cholesterol = st.number_input("Cholesterol", min_value=50, max_value=600, value=200)

# FIXED: FastingBS as NUMERICAL (0 or 1)
fasting_bs = st.selectbox("Fasting Blood Sugar", [0, 1]) 
rest_ecg = st.selectbox("RestingECG", ["Normal", "ST", "LVH"])


max_hr = st.number_input("Maximum Heart Rate", min_value=50, max_value=220, value=130)
exercise_angina = st.selectbox("Exercise Angina", ["Yes", "No"])

oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, value=1.0)


# Ordinal Categories
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# -----------------------------
# Convert to DataFrame
# -----------------------------
input_data = pd.DataFrame({
    "Age": [age],
    "RestingBP": [resting_bp],
    "Cholesterol": [cholesterol],
    "FastingBS": [fasting_bs],      
    "MaxHR": [max_hr],
    "Oldpeak": [oldpeak],
    "Sex": [sex],
    "RestingECG": [rest_ecg],
    "ExerciseAngina": [exercise_angina],
    "ChestPainType": [chest_pain],
    "ST_Slope": [st_slope]
})

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Heart Disease"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"❌ High Risk of Heart Disease\nProbability: {probability:.2f}")
    else:
        st.success(f"✔ No Heart Disease\nProbability: {probability:.2f}")




