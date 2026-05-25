import streamlit as st
import requests

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Financial Risk Predictor",
    page_icon="💰",
    layout="wide"
)

# ======================================================
# CUSTOM CSS
# ======================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.stButton > button {
    width: 100%;
    background-color: #1f77b4;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 50px;
    border: none;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #125d98;
    color: white;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    margin-top: 20px;
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

# ======================================================
# TITLE
# ======================================================

st.title("💰 Financial Risk Prediction System")

st.write("Enter customer financial details below to predict risk level.")

st.markdown("---")

# ======================================================
# INPUT FIELDS
# ======================================================

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    annual_income = st.number_input(
        "Annual Income",
        min_value=10000.0,
        max_value=1000000.0,
        value=50000.0
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300.0,
        max_value=850.0,
        value=650.0
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=1000.0,
        max_value=1000000.0,
        value=20000.0
    )

    interest_rate = st.number_input(
        "Interest Rate (%)",
        min_value=1.0,
        max_value=50.0,
        value=10.0
    )

    debt_to_income_ratio = st.slider(
        "Debt To Income Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.3
    )

    savings_balance = st.number_input(
        "Savings Balance",
        min_value=0.0,
        max_value=1000000.0,
        value=10000.0
    )

with col2:

    work_experience_years = st.number_input(
        "Work Experience (Years)",
        min_value=0,
        max_value=50,
        value=5
    )

    loan_duration_months = st.number_input(
        "Loan Duration (Months)",
        min_value=1,
        max_value=360,
        value=24
    )

    monthly_expenses = st.number_input(
        "Monthly Expenses",
        min_value=0.0,
        max_value=100000.0,
        value=5000.0
    )

    employment_stability_years = st.number_input(
        "Employment Stability (Years)",
        min_value=0,
        max_value=50,
        value=5
    )

    previous_default_count = st.number_input(
        "Previous Defaults",
        min_value=0,
        max_value=20,
        value=0
    )

    financial_literacy_score = st.slider(
        "Financial Literacy Score",
        min_value=1,
        max_value=100,
        value=50
    )

# ======================================================
# PREDICTION BUTTON
# ======================================================

if st.button("Predict Financial Risk"):

    # ==================================================
    # CREATE INPUT DATA
    # ==================================================

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

    # ==================================================
    # RENDER BACKEND URL
    # ==================================================

    API_URL = "https://financial-risk-prediction-system-2.onrender.com/predict"

    # ==================================================
    # SEND REQUEST TO FASTAPI BACKEND
    # ==================================================

    try:

        with st.spinner("Predicting financial risk..."):

            response = requests.post(
                API_URL,
                json=data,
                timeout=30
            )

        st.markdown("---")

        # ==================================================
        # SUCCESS RESPONSE
        # ==================================================

        if response.status_code == 200:

            result = response.json()

            prediction = result.get("prediction", "Unknown")
            confidence = result.get("confidence", 0)

            if prediction.lower() == "low risk":

                st.markdown(f"""
                <div class="result-box low-risk">
                    ✅ LOW FINANCIAL RISK <br><br>
                    Confidence: {confidence}%
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown(f"""
                <div class="result-box high-risk">
                    ⚠️ HIGH FINANCIAL RISK <br><br>
                    Confidence: {confidence}%
                </div>
                """, unsafe_allow_html=True)

        # ==================================================
        # SERVER ERROR
        # ==================================================

        else:

            st.error(f"Server Error: {response.status_code}")

            st.write(response.text)

    # ==================================================
    # TIMEOUT ERROR
    # ==================================================

    except requests.exceptions.Timeout:

        st.error("Request Timeout. Render backend may be sleeping.")

    # ==================================================
    # CONNECTION ERROR
    # ==================================================

    except requests.exceptions.ConnectionError:

        st.error("Failed to connect to backend server.")

    # ==================================================
    # OTHER ERRORS
    # ==================================================

    except Exception as e:

        st.error("Unexpected Error Occurred")

        st.write(e)
