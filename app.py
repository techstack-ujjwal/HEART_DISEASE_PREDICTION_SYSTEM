import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessing objects
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# Numeric columns used in scaler
numeric_cols = ['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']

# UI
st.title("❤️ Heart Disease Prediction System")
st.caption("HV PROJECT")

st.write("Please enter patient details:")

age = st.slider("Age", 1, 100, 25)

sex = st.selectbox(
    "Sex",
    ["Male","Female"]
)

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ATA","NAP","TA","ASY"]
)

resting_bp = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    80,200,120
)

cholesterol = st.number_input(
    "Cholesterol (mg/dl)",
    100,600,200
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    ["0","1"]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal","ST","LVH"]
)

max_hr = st.slider(
    "Maximum Heart Rate",
    60,220,150
)

exercise_angina = st.selectbox(
    "Exercise Induced Angina",
    ["Y","N"]
)

oldpeak = st.number_input(
    "Oldpeak (ST depression)",
    0.0,6.0,1.0
)

st_slope = st.selectbox(
    "ST Slope",
    ["Up","Flat","Down"]
)

# Prediction
if st.button("Predict"):

    raw_input = {
    "Age": age,
    "RestingBP": resting_bp,
    "Cholesterol": cholesterol,
    "FastingBS": int(fasting_bs),
    "MaxHR": max_hr,
    "Oldpeak": oldpeak,

    "Sex_M": 1 if sex == "Male" else 0,

    "ChestPainType_ATA": 1 if chest_pain == "ATA" else 0,
    "ChestPainType_NAP": 1 if chest_pain == "NAP" else 0,
    "ChestPainType_TA": 1 if chest_pain == "TA" else 0,

    "RestingECG_Normal": 1 if resting_ecg == "Normal" else 0,
    "RestingECG_ST": 1 if resting_ecg == "ST" else 0,

    "ExerciseAngina_Y": 1 if exercise_angina == "Y" else 0,

    "ST_Slope_Flat": 1 if st_slope == "Flat" else 0,
    "ST_Slope_Up": 1 if st_slope == "Up" else 0
}



    # Convert to dataframe
    input_df = pd.DataFrame([raw_input])

    # Match training columns exactly
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    # Scale ONLY numeric features
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Predict
    prediction = model.predict(input_df)[0]

    # Output result
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.info("⚠️ This is a machine learning prediction and not a medical diagnosis.")

    prob = model.predict_proba(input_df)[0][1]
    st.write("Heart Disease Probability:", round(prob*100,2), "%")
