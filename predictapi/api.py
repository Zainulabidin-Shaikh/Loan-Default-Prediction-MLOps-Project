from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import os

# --- Feature engineering function (copy this from training!) ---
def add_comprehensive_features(df):
    df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1e-8)
    df['monthly_payment'] = df['loan_amount'] / (df['term_months'] + 1e-8)
    df['payment_to_income_ratio'] = (df['monthly_payment'] * 12) / (df['annual_income'] + 1e-8)
    df['employment_risk'] = (df['employment_length'] < 2).astype(int)
    df['young_borrower'] = (df['age'] < 30).astype(int)
    df['senior_borrower'] = (df['age'] > 60).astype(int)
    df['experienced_worker'] = (df['employment_length'] > 10).astype(int)
    df['high_credit_score'] = (df['credit_score'] > 750).astype(int)
    df['low_credit_score'] = (df['credit_score'] < 580).astype(int)
    df['high_interest'] = (df['interest_rate'] > df['interest_rate'].median()).astype(int)
    df['multiple_delinquencies'] = (df['delinquency_2yrs'] > 1).astype(int)
    df['many_open_accounts'] = (df['num_open_acc'] > df['num_open_acc'].median()).astype(int)
    if 'total_acc' in df.columns and 'num_open_acc' in df.columns:
        df['account_utilization'] = df['num_open_acc'] / (df['total_acc'] + 1e-8)
    df['credit_score_binned'] = pd.cut(
        df['credit_score'],
        bins=[0, 580, 670, 740, 850],
        labels=['Poor', 'Fair', 'Good', 'Excellent'],
        include_lowest=True
    )
    try:
        df['income_binned'] = pd.qcut(df['annual_income'], q=5, labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'], duplicates='drop')
    except Exception:
        df['income_binned'] = 'Medium'
    try:
        df['loan_amount_binned'] = pd.qcut(df['loan_amount'], q=5, labels=['XS', 'S', 'M', 'L', 'XL'], duplicates='drop')
    except Exception:
        df['loan_amount_binned'] = 'M'
    risk_factors = ['employment_risk', 'young_borrower', 'high_interest', 
                   'multiple_delinquencies', 'many_open_accounts', 'low_credit_score']
    df['risk_score'] = df[risk_factors].sum(axis=1)
    positive_factors = ['high_credit_score', 'experienced_worker']
    df['positive_score'] = df[positive_factors].sum(axis=1)
    df['net_risk_score'] = df['risk_score'] - df['positive_score']
    df['income_age_interaction'] = df['annual_income'] * df['age']
    df['credit_employment_interaction'] = df['credit_score'] * df['employment_length']
    return df


# --- Input schema with underscores ---
class LoanInput(BaseModel):
    age: int
    annual_income: float
    employment_length: int
    home_ownership: str
    purpose: str
    loan_amount: float
    term_months: int
    interest_rate: float
    dti: float
    credit_score: int
    delinquency_2yrs: int
    num_open_acc: int

app = FastAPI()
MODEL_PATH = os.path.abspath("exported_model/best_loan_model")
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = mlflow.sklearn.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Model load error: {e}")
        model = None

@app.get("/")
def root():
    return {"message": "Loan Default Prediction API running. See /docs for usage."}

@app.get("/health")
def health():
    return {"model_loaded": model is not None}

@app.post("/predict")
def predict_loan(input: LoanInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    df = pd.DataFrame([input.dict()])
    # Feature engineering step is critical!
    df = add_comprehensive_features(df)
    # categorical bin fields must be type string/object (not category) if pipeline expects so
    for col in ['credit_score_binned', 'income_binned', 'loan_amount_binned']:
        if col in df.columns:
            df[col] = df[col].astype(str)
    try:
        proba = model.predict_proba(df).tolist()[0][1]
        prediction = model.predict(df).tolist()[0]
        return {
            "default_probability": proba,
            "default_prediction": int(prediction)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
