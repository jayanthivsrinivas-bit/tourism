import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# -----------------------------
# Load model from Hugging Face
# -----------------------------
model_file = hf_hub_download(
    repo_id="Fitjv/tourism-model",
    filename="tourism_model_xgb.joblib"
)
model = joblib.load(model_file)

st.title("Tourism Customer Prediction")
st.write("Predict whether a customer will take the offered product.")

# -----------------------------
# Input form
# -----------------------------
with st.form("customer_form"):
    Age = st.number_input("Age", min_value=18, max_value=100, value=30)
    MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=1000000, value=50000)
    DurationOfPitch = st.number_input("Duration Of Pitch (minutes)", min_value=1, max_value=120, value=10)
    NumberOfTrips = st.number_input("Number of Trips", min_value=0, max_value=50, value=2)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Occupation = st.selectbox("Occupation", ["Salaried", "Business", "Self-Employed", "Other"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([{
        "Age": Age,
        "MonthlyIncome": MonthlyIncome,
        "DurationOfPitch": DurationOfPitch,
        "NumberOfTrips": NumberOfTrips,
        "Gender": Gender,
        "Occupation": Occupation,
        "MaritalStatus": MaritalStatus
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"Prediction: {prediction}")
