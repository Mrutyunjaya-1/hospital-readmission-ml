import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
@st.cache_resource
def load_model():
    with open("best_rf_model.joblib", "rb") as f:
        model = joblib.load(f)
    with open("feature_columns.joblib", "rb") as f:
        feature_cols = joblib.load(f)
    return model, feature_cols

model, feature_cols = load_model()

st.set_page_config(page_title="Hospital Readmission Predictor", layout="centered")
st.title("🏥 Hospital Readmission Risk Predictor")

def user_input_features():
    # --- Your existing UI inputs ---
    age = st.selectbox("Age Group", [
        "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", 
        "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
    ])
    gender = st.selectbox("Gender", ["Male", "Female"])
    race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other", "Unknown"])
    admission_type_id = st.selectbox("Admission Type ID", [1, 2, 3, 4, 5, 6, 7, 8])
    discharge_disposition_id = st.selectbox("Discharge Disposition ID", list(range(1, 31)))
    admission_source_id = st.selectbox("Admission Source ID", list(range(1, 26)))
    time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 3)
    num_lab_procedures = st.slider("Number of Lab Procedures", 0, 150, 40)
    num_procedures = st.slider("Number of Procedures", 0, 6, 1)
    num_medications = st.slider("Number of Medications", 1, 80, 10)
    number_outpatient = st.slider("Outpatient Visits (past year)", 0, 50, 0)
    number_emergency = st.slider("Emergency Visits (past year)", 0, 50, 0)
    number_inpatient = st.slider("Inpatient Visits (past year)", 0, 50, 0)
    number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 5)

    insulin = st.selectbox("Insulin", ["No", "Up", "Down", "Steady"])
    metformin = st.selectbox("Metformin", ["No", "Up", "Down", "Steady"])
    change = st.selectbox("Medication Change", ["No", "Ch"])
    diabetesMed = st.selectbox("Diabetes Medication", ["Yes", "No"])

    # Only the fields you collect from UI
    data = {
        "race": race,
        "gender": gender,
        "age": age,
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
        "insulin": insulin,
        "metformin": metformin,
        "change": change,
        "diabetesMed": diabetesMed
    }

    df = pd.DataFrame([data])

    # 🔧 Align to training schema (add missing columns)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Reorder columns to match training
    df = df[feature_cols]
    return df

input_df = user_input_features()

st.subheader("Input Summary")
st.dataframe(input_df.head())

if st.button("Predict Readmission Risk"):
    prob = model.predict_proba(input_df)[0][1]
    pred = int(prob > 0.5)

    st.metric("Readmission Risk (Probability)", f"{prob:.2%}")

    if pred == 1:
        st.error("⚠️ High Risk: Patient likely to be readmitted within 30 days.")
    else:
        st.success("✅ Low Risk: Patient unlikely to be readmitted within 30 days.")
