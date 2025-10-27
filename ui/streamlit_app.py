import streamlit as st
import requests

st.title("Loan Default Prediction")

st.write(
    "Enter applicant info and get model prediction from FastAPI backend."
)

fields = {
    "age": st.number_input("Age", min_value=18, max_value=100, value=32),
    "annual_income": st.number_input("Annual Income", value=60000.0),
    "employment_length": st.number_input("Employment Length (years)", min_value=0, max_value=50, value=3),
    "home_ownership": st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"], index=0),
    "purpose": st.text_input("Purpose", "creditcard"),
    "loan_amount": st.number_input("Loan Amount", value=15000.0),
    "term_months": st.selectbox("Term (months)", [12, 24, 36, 60], index=2),
    "interest_rate": st.number_input("Interest Rate", min_value=0.0, max_value=100.0, value=12.5),
    "dti": st.number_input("Debt-to-Income Ratio (DTI)", value=20.3),
    "credit_score": st.number_input("Credit Score", min_value=300, max_value=850, value=720),
    "delinquency_2yrs": st.number_input("Delinquencies in 2 Years", min_value=0, max_value=10, value=0),
    "num_open_acc": st.number_input("Number of Open Accounts", min_value=0, max_value=30, value=6)
}

if st.button("Predict"):
    resp = requests.post("http://localhost:9000/predict", json=fields)
    if resp.status_code == 200:
        result = resp.json()
        pred = result["default_prediction"]
        prob = result["default_probability"]
        st.success(f"Default Probability: {prob:.2%}")
        st.info("Prediction: " + ("Yes (Will Default)" if pred else "No (Safe)"))
    else:
        st.error(f"API Error: {resp.status_code}\n{resp.text}")
