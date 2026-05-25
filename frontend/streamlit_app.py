import streamlit as st
import requests

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Financial Risk Predictor",
    page_icon="💰",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.stButton>button {
    width: 100%;
    background-color: #1f77b4;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 50px;
    border: none;
}

.stButton>button:hover {
    background-color: #125d98;
}

.result-box {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

.low-risk {
    background-color: #d4edda;
    color: #155724;
}

.high-risk {
    background-color: #f8d7da;
    color: #721c24;
}

</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================

st.title("💰 Financial Risk Prediction System")

st.write("Fill customer financial details below.")

# =========================
# INPUTS
# =========================

col1, col2 = st.columns(2)

with col1:

    age = st.number_input("Age", 18, 100, 30)

    annual_income = st.number_input(
        "Annual Income",
        10000.0,
        1000000.0,
        50000.0
    )

    credit_score = st.number_input(
        "Credit Score",
        300.0,
        850.0,
        650.0
    )

    loan_amount = st.number_input(
        "Loan Amount",
        1000.0,
        1000000.0,
        20000.0
    )

    interest_rate = st.number_input(
        "Interest Rate (%)",
        1.0,
        50.0,
        10.0
    )

    debt_to_income_ratio = st.slider(
        "Debt To Income Ratio",
        0.0,
        1.0,
        0.3
    )

    savings_balance = st.number_input(
        "Savings Balance",
        0.0,
        1000000.0,
        10000.0
    )

with col2:

    work_experience_years = st.number_input(
        "Work Experience",
        0,
        50,
        5
    )

    loan_duration_months = st.number_input(
        "Loan Duration (Months)",
        1,
        360,
        24
    )

    monthly_expenses = st.number_input(
        "Monthly Expenses",
        0.0,
        100000.0,
        5000.0
    )

    employment_stability_years = st.number_input(
        "Employment Stability",
        0,
        50,
        5
    )

    previous_default_count = st.number_input(
        "Previous Defaults",
        0,
        20,
        0
    )

    financial_literacy_score = st.slider(
        "Financial Literacy Score",
        1,
        100,
        50
    )

# =========================
# PREDICT BUTTON
# =========================

if st.button("Predict Financial Risk"):

    data = {
        "age": age,
        "annual_income": annual_income,
        "education_years": 12,
        "work_experience_years": work_experience_years,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "loan_duration_months": loan_duration_months,
        "interest_rate": interest_rate,
        "debt_to_income_ratio": debt_to_income_ratio,
        "monthly_expenses": monthly_expenses,
        "savings_balance": savings_balance,
        "investment_balance": 5000,
        "number_of_dependents": 2,
        "employment_stability_years": employment_stability_years,
        "housing_status_score": 3,
        "marital_status_score": 2,
        "health_risk_index": 0.5,
        "insurance_coverage_score": 3,
        "spending_score": 50,
        "online_activity_score": 50,
        "device_usage_hours": 5,
        "location_risk_index": 0.5,
        "previous_default_count": previous_default_count,
        "financial_literacy_score": financial_literacy_score
    }

    try:

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=data
        )

        result = response.json()

        prediction = result["prediction"]
        confidence = result["confidence"]

        st.markdown("---")

        if prediction == "Low Risk":

            st.markdown(f'''
            <div class="result-box low-risk">
            ✅ LOW FINANCIAL RISK <br><br>
            Confidence: {confidence}%
            </div>
            ''', unsafe_allow_html=True)

        else:

            st.markdown(f'''
            <div class="result-box high-risk">
            ⚠️ HIGH FINANCIAL RISK <br><br>
            Confidence: {confidence}%
            </div>
            ''', unsafe_allow_html=True)

    except Exception as e:

        st.error("FastAPI server is not running.")
        st.write(e)