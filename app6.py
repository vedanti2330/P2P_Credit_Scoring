import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page configuration
st.set_page_config(page_title="P2P Credit risk Intelligence", layout="wide")

# Professional Header
st.markdown("""
    <style>
    .main-header {
        background-color: #1e293b;
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        border-bottom: 5px solid #3b82f6;
        margin-bottom: 2rem;
    }
    </style>
    <div class="main-header">
        <h1>🛡️ P2P Lending: Advanced Credit Scoring</h1>
        <p>Predictive Analytics for Default Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    # This must be the PIPELINE object saved from your notebook
    return joblib.load('Best_final_model2.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Input Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Borrower Profile")
    annual_inc = st.number_input("Annual Income ($)", min_value=0, value=50000)
    emp_length_raw = st.selectbox("Employment Length", 
                                 ['< 1 year', '1 year', '2 years', '3 years', '4 years', 
                                  '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'])
    home_ownership = st.selectbox("Home Ownership", ['RENT', 'MORTGAGE', 'OWN', 'OTHER'])
    verification_status = st.selectbox("Verification Status", ['Not Verified', 'Source Verified', 'Verified'])
    dti = st.number_input("DTI Ratio", min_value=0.0, max_value=100.0, value=15.0)

with col2:
    st.subheader("💰 Loan Details")
    loan_amnt = st.number_input("Loan Amount ($)", min_value=500, value=10000)
    term_raw = st.selectbox("Term", [' 36 months', ' 60 months'])
    int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=12.0)
    installment = st.number_input("Monthly Installment ($)", min_value=0.0, value=320.0)
    grade = st.selectbox("Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    purpose = st.selectbox("Purpose", ['debt_consolidation', 'credit_card', 'home_improvement', 'other', 'major_purchase', 'small_business'])

st.divider()

if st.button("🚀 Run Credit Risk Analysis", use_container_width=True):
    try:
        # --- DATA PRE-PROCESSING ---
        # 1. Convert Term to numeric
        term_numeric = int(term_raw.replace(' months', '').strip())
        
        # 2. Convert Emp Length to numeric
        if '< 1' in emp_length_raw:
            emp_numeric = 0
        elif '10+' in emp_length_raw:
            emp_numeric = 10
        else:
            emp_numeric = int(''.join(filter(str.isdigit, emp_length_raw)))

        # 3. Construct DataFrame with EXACTLY 11 columns in training order
        input_data = pd.DataFrame({
            'loan_amnt': [loan_amnt],
            'term': [term_numeric],
            'int_rate': [int_rate],
            'installment': [installment],
            'grade': [grade],
            'annual_inc': [annual_inc],
            'dti': [dti],
            'emp_length': [emp_numeric],
            'home_ownership': [home_ownership],
            'verification_status': [verification_status],
            'purpose': [purpose]
        })

        # Predict using the Pipeline
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        # Results Display
        st.subheader("📊 Model Assessment")
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            status = "CHARGED OFF" if prediction == 1 else "FULLY PAID"
            st.metric("Predicted Outcome", status)
        
        with res_col2:
            st.metric("Default Probability", f"{probability:.2%}")
        
        with res_col3:
            risk_lvl = "HIGH RISK" if probability > 0.5 else "LOW RISK"
            risk_icon = "🔴" if probability > 0.5 else "🟢"
            st.metric("Risk Level", f"{risk_icon} {risk_lvl}")

        if prediction == 1:
            st.error(f"**High Risk:** Probability of default is {probability:.1%}.")
        else:
            st.success(f"**Low Risk:** Likelihood of full repayment is high.")

    except Exception as e:
        st.error(f"❌ **Error:** {e}")
