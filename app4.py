import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page configuration
st.set_page_config(page_title="P2P Credit Risk Intelligence", layout="wide")

# Custom CSS for a clean financial dashboard look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e1e4e8; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    # Loading the model specified by the user
    return joblib.load('Best_final_model.pkl')

# Attempt to load the model and handle the error if it's just an array
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Professional Header
st.title("🛡️ P2P Lending: Credit Scoring & Default Prediction")
st.markdown("""
    ### Machine Learning Risk Analytics
    Input borrower data below to assess the probability of loan default (**Charged Off**) versus successful repayment (**Fully Paid**). 
    The model evaluates creditworthiness based on financial history and loan characteristics.
""")
st.divider()

# Input UI Sections
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Borrower Profile")
    annual_inc = st.number_input("Annual Income ($)", min_value=0, value=55000)
    emp_length = st.selectbox("Employment Length", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], format_func=lambda x: f"{x}+ years" if x==10 else (f"< 1 year" if x==0 else f"{x} years"))
    home_ownership = st.selectbox("Home Ownership", ['RENT', 'MORTGAGE', 'OWN', 'OTHER'])
    dti = st.number_input("Debt-to-Income (DTI) Ratio", min_value=0.0, max_value=100.0, value=15.0)

with col2:
    st.subheader("💰 Loan Specifications")
    loan_amnt = st.number_input("Loan Amount ($)", min_value=500, value=10000)
    term = st.selectbox("Term (Months)", [36, 60])
    int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=12.5)
    purpose = st.selectbox("Purpose", ['debt_consolidation', 'credit_card', 'home_improvement', 'other', 'major_purchase', 'small_business'])

# Prediction Section
if st.button("🚀 Analyze Credit Risk", use_container_width=True):
    try:
        # Construct input DataFrame with columns exactly as trained in the notebook
        # Note: 'installment' and 'grade' were used in the notebook; we estimate installment here
        # monthly_rate = (int_rate / 100) / 12
        # installment = (loan_amnt * monthly_rate) / (1 - (1 + monthly_rate)**(-term))
        
        input_df = pd.DataFrame({
            'loan_amnt': [loan_amnt],
            'term': [term],
            'int_rate': [int_rate],
            'annual_inc': [annual_inc],
            'dti': [dti],
            'emp_length': [emp_length],
            'home_ownership': [home_ownership],
            'verification_status': ['Verified'], # Defaulting for prediction
            'purpose': [purpose],
            'grade': ['B'], # Defaulting for prediction as it's a model requirement
            'installment': [loan_amnt / term] # simplified estimate
        })

        # Check if the object has the predict method
        if not hasattr(model, 'predict'):
            st.error("❌ **Prediction Error:** The loaded file 'Best_final_model.pkl' is a NumPy array, not a model object. Please re-run your notebook and save the actual model variable (e.g., `joblib.dump(final_pipeline, 'Best_final_model.pkl')`).")
        else:
            # 0 = Fully Paid, 1 = Charged Off
            prediction = model.predict(input_df)[0]
            # Probability of Default (Class 1)
            prob_default = model.predict_proba(input_df)[0][1] 
            
            st.divider()
            st.subheader("📊 Assessment Summary")
            
            m_col1, m_col2, m_col3 = st.columns(3)
            
            with m_col1:
                result_text = "CHARGED OFF (Default)" if prediction == 1 else "FULLY PAID"
                st.metric("Predicted Outcome", result_text)
                
            with m_col2:
                st.metric("Default Probability", f"{prob_default:.2%}")
                
            with m_col3:
                risk_cat = "HIGH RISK" if prob_default > 0.5 else "LOW RISK"
                risk_icon = "🔴" if prob_default > 0.5 else "🟢"
                st.metric("Risk Level", f"{risk_icon} {risk_cat}")

            # Detailed Output
            if prediction == 1:
                st.error(f"**Warning:** The model predicts a high risk of default ({prob_default:.1%} probability).")
            else:
                st.success(f"**Confidence:** High likelihood of full repayment (Probability of default: {prob_default:.1%}).")

    except Exception as e:
        st.error(f"❌ **System Error:** {e}")
        st.info("Ensure the input features match the model's training requirements (Columns: loan_amnt, term, int_rate, installment, grade, annual_inc, dti, emp_length, home_ownership, verification_status, purpose).")
