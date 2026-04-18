import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Setup
st.set_page_config(page_title="P2P Credit Risk Analytics", layout="wide")

# Professional Header
st.markdown("""
    <div style="background-color:#0e1117;padding:20px;border-radius:10px;border-bottom: 5px solid #ff4b4b">
        <h1 style="color:white;text-align:center;">🛡️ P2P Lending Credit Risk Intelligence</h1>
        <p style="color:#fafafa;text-align:center;">Predictive Analytics for Default Assessment & Loan Scoring</p>
    </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    # Loading your specific model file
    return joblib.load('Best_final_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load 'Best_final_model.pkl'. Ensure the file is in the same folder as app.py.")
    st.stop()

st.write("") # Spacing

# Sidebar for Inputs
st.sidebar.header("📋 Borrower Application Data")

def get_user_input():
    loan_amnt = st.sidebar.number_input("Loan Amount ($)", value=10000)
    term = st.sidebar.selectbox("Term", [' 36 months', ' 60 months'])
    int_rate = st.sidebar.number_input("Interest Rate (%)", value=12.0)
    annual_inc = st.sidebar.number_input("Annual Income ($)", value=50000)
    emp_length = st.sidebar.selectbox("Employment Length", ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'])
    home_ownership = st.sidebar.selectbox("Home Ownership", ['RENT', 'MORTGAGE', 'OWN', 'OTHER'])
    verification_status = st.sidebar.selectbox("Verification Status", ['Not Verified', 'Source Verified', 'Verified'])
    purpose = st.sidebar.selectbox("Purpose", ['debt_consolidation', 'credit_card', 'home_improvement', 'other', 'major_purchase', 'small_business'])
    
    # IMPORTANT: Include any other columns you kept in your notebook (e.g., dti, installment)
    # If you dropped them in the notebook, they must not be here.
    data = {
        'loan_amnt': loan_amnt,
        'term': term,
        'int_rate': int_rate,
        'annual_inc': annual_inc,
        'emp_length': emp_length,
        'home_ownership': home_ownership,
        'verification_status': verification_status,
        'purpose': purpose
    }
    return pd.DataFrame(data, index=[0])

input_df = get_user_input()

# Display Input Summary
st.subheader("Summary of Input Features")
st.dataframe(input_df)

# Prediction Logic
if st.button("Analyze Risk Profile"):
    try:
        # 1. Check for 'predict' attribute (Fixes the numpy error)
        if not hasattr(model, 'predict'):
            st.error("🚨 **Model Object Error:** The file loaded is a list of numbers (array), not the model. Re-save your model in the notebook using `joblib.dump(model_variable, 'Best_final_model.pkl')`.")
        else:
            # 2. Perform Prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
            
            # Map prediction to user-friendly text
            # Assuming 0 = Fully Paid, 1 = Charged Off
            result = "CHARGED OFF (Default)" if prediction == 1 else "FULLY PAID"
            prob_val = probability[1] if len(probability) > 1 else 0.0

            # 3. Output UI
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error(f"### Predicted Status: {result}")
                else:
                    st.success(f"### Predicted Status: {result}")
            
            with col2:
                st.metric("Default Probability", f"{prob_val:.2%}")

            # Visualization
            st.progress(prob_val)
            if prob_val > 0.5:
                st.warning("⚠️ This application shows high indicators of potential default.")
            else:
                st.info("✅ This application shows strong indicators for full repayment.")

    except Exception as e:
        st.error(f"**Feature Mismatch Error:** {e}")
        st.info("Check if your notebook used more columns (like 'dti' or 'grade') that are missing here.")
