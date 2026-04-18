import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page configuration
st.set_page_config(page_title="P2P Credit Risk Intelligence", layout="wide")

# Custom CSS for Professional Header
st.markdown("""
    <style>
    .header-box {
        background-color: #1e293b;
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        border-bottom: 5px solid #3b82f6;
        margin-bottom: 25px;
    }
    </style>
    <div class="header-box">
        <h1>🛡️ P2P Lending: Credit Risk & Default Intelligence</h1>
        <p>Predictive Analytics for Peer-to-Peer Loan Assessment</p>
    </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    # Loading the specific model you named
    return joblib.load('Best_final_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading 'Best_final_model.pkl': {e}")
    st.stop()

# Input UI in Columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Borrower Details")
    annual_inc = st.number_input("Annual Income ($)", min_value=0, value=50000)
    emp_length = st.selectbox("Employment Length", ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'])
    home_ownership = st.selectbox("Home Ownership", ['RENT', 'MORTGAGE', 'OWN', 'OTHER'])
    verification_status = st.selectbox("Verification Status", ['Not Verified', 'Source Verified', 'Verified'])
    dti = st.number_input("Debt-to-Income (DTI) Ratio", min_value=0.0, max_value=100.0, value=15.0)

with col2:
    st.subheader("💰 Loan Financials")
    loan_amnt = st.number_input("Loan Amount ($)", min_value=500, value=10000)
    term = st.selectbox("Term", [' 36 months', ' 60 months'])
    int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=12.0)
    installment = st.number_input("Monthly Installment ($)", min_value=0.0, value=300.0)
    grade = st.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    purpose = st.selectbox("Purpose", ['debt_consolidation', 'credit_card', 'home_improvement', 'other', 'major_purchase', 'small_business'])

st.divider()

if st.button("🚀 Analyze Default Probability", use_container_width=True):
    try:
        # Construct DataFrame with exactly the 11 columns your model expects
        # The names MUST match the training columns exactly
        input_data = pd.DataFrame({
            'loan_amnt': [loan_amnt],
            'term': [term],
            'int_rate': [int_rate],
            'installment': [installment],
            'grade': [grade],
            'emp_length': [emp_length],
            'home_ownership': [home_ownership],
            'annual_inc': [annual_inc],
            'verification_status': [verification_status],
            'purpose': [purpose],
            'dti': [dti]
        })

        # Logic to handle both Model objects and accidental Array saves
        if not hasattr(model, 'predict'):
            st.error("❌ **Object Error:** The file 'Best_final_model.pkl' is a result array, not the actual model. Please re-save the model object in your notebook using: `joblib.dump(your_model_name, 'Best_final_model.pkl')`.")
        else:
            # Prediction (0 = Fully Paid, 1 = Charged Off/Default)
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1] # Probability of Class 1

            st.subheader("📋 Risk Assessment Result")
            res1, res2, res3 = st.columns(3)
            
            with res1:
                final_status = "DEFAULT / CHARGED OFF" if prediction == 1 else "FULLY PAID"
                st.metric("Predicted Outcome", final_status)
            
            with res2:
                st.metric("Probability of Default", f"{probability:.2%}")
            
            with res3:
                risk_level = "HIGH RISK" if probability > 0.5 else "LOW RISK"
                risk_icon = "🔴" if probability > 0.5 else "🟢"
                st.metric("Risk Status", f"{risk_icon} {risk_level}")

            if prediction == 1:
                st.error(f"**Warning:** High risk of default detected ({probability:.1%}). Financial caution advised.")
            else:
                st.success(f"**Approval Insight:** Strong profile for successful repayment (Probability of default: {probability:.1%}).")

    except Exception as e:
        st.error(f"❌ **Technical Error:** {e}")
        st.info("Ensure that 'Best_final_model.pkl' was saved using the exact same 11 columns used here.")
