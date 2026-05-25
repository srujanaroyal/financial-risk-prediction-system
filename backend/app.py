from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('svm_financial_risk_model.pkl')
scaler = joblib.load('scaler.pkl')

app = FastAPI()

class RiskInput(BaseModel):
    age: int
    annual_income: float
    education_years: int
    work_experience_years: int
    credit_score: float
    loan_amount: float
    loan_duration_months: int
    interest_rate: float
    debt_to_income_ratio: float
    monthly_expenses: float
    savings_balance: float
    investment_balance: float
    number_of_dependents: int
    employment_stability_years: int
    housing_status_score: int
    marital_status_score: int
    health_risk_index: float
    insurance_coverage_score: int
    spending_score: int
    online_activity_score: int
    device_usage_hours: float
    location_risk_index: float
    previous_default_count: int
    financial_literacy_score: int

@app.get('/')
def home():
    return {'message': 'Financial Risk API Running'}

@app.post('/predict')
def predict(data: RiskInput):

    input_data = np.array([[
        data.age,
        data.annual_income,
        data.education_years,
        data.work_experience_years,
        data.credit_score,
        data.loan_amount,
        data.loan_duration_months,
        data.interest_rate,
        data.debt_to_income_ratio,
        data.monthly_expenses,
        data.savings_balance,
        data.investment_balance,
        data.number_of_dependents,
        data.employment_stability_years,
        data.housing_status_score,
        data.marital_status_score,
        data.health_risk_index,
        data.insurance_coverage_score,
        data.spending_score,
        data.online_activity_score,
        data.device_usage_hours,
        data.location_risk_index,
        data.previous_default_count,
        data.financial_literacy_score
    ]])

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0].max()

    result = 'High Risk' if prediction == 1 else 'Low Risk'

    return {
        'prediction': result,
        'confidence': round(float(probability) * 100, 2)
    }
