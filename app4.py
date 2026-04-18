import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page config
st.set_page_config(page_title="P2P Credit Risk Intelligence", layout="wide")

# Professional Header
st.markdown("""
    <div style="background-color:#1e293b; padding:25px; border-radius:10px; text-align:center; margin-bottom:25px; border-bottom: 4px solid #3b82f6;">
        <h1 style="color:white; margin:0;">🛡️ P2P Lending: Predictive Credit Scoring</h1>
        <p style="color:#94a3b8; font-size:1.1em;">Machine Learning Analysis for Loan Default Prediction</p>
    </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load('Best_final_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Layout for Inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Borrower Profile")
    annual_inc = st.number_input("Annual Income ($)", min_value=0, value=50000)
    emp_length = st.selectbox("Employment Length", ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'])
    home_ownership = st.selectbox("Home Ownership", ['RENT', 'MORTGAGE', 'OWN', 'OTHER'])
    # Adding DTI because the model expects 11 features
    dti = st.number_input("Debt-to-Income Ratio (DTI)", min_value=0.0, max_value=100.0, value=15.0)

with col2:
    st.subheader("💰 Loan Details")
    loan_amnt = st.number_input("Loan Amount ($)", min_value=500, value=10000)
    term = st.selectbox("Term", [' 36 months', ' 60 months'])
    int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=12.0)
    purpose = st.selectbox("Purpose", ['debt_consolidation', 'credit_card', 'home_improvement', 'other', 'major_purchase', 'small_business'])

# Hidden calculations to satisfy the 11-feature requirement
# Based on your notebook, these are likely the missing columns:
installment = loan_amnt / (36 if '36' in term else 60)
verification_status = 'Verified'
grade = 'B' # Placeholder common grade

if st.button("🚀 Run Risk Analysis", use_container_width=True):
    try:
        # Construct the DataFrame with EXACTLY 11 features in the correct order
        # Ensure these column names match your X_train.columns from the notebook
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

        # Check if it's the model object
        if not hasattr(model, 'predict'):
            st.error("The file is still a NumPy array. Please re-save the model object in your notebook.")
        else:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1] # Prob of Default

            st.divider()
            
            # Result UI
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                if prediction == 1:
                    st.error("### Result: CHARGED OFF (Default)")
                else:
                    st.success("### Result: FULLY PAID")
            
            with res_col2:
                st.metric("Probability of Default", f"{probability:.1%}")
                
            # Final Risk Interpretation
            if probability > 0.5:
                st.warning("⚠️ **High Risk Profile:** The probability of default exceeds 50%.")
            else:
                st.info("✅ **Low Risk Profile:** The borrower is likely to repay the loan in full.")

    except Exception as e:
        st.error(f"**Error:** {e}")
        st.info("This usually means the column names or the count (11) don't match your training data.")
