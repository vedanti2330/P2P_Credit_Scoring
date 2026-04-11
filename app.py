import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the trained model pipeline
# Make sure the file 'loan_model.joblib' is in the same folder as this script
try:
    model = joblib.load('loan_model.joblib')
except FileNotFoundError:
    st.error("Model file 'loan_model.joblib' not found. Please ensure the model is trained and saved.")

# 2. Set up the UI header
st.set_page_config(page_title="Loan Default Prediction", layout="centered")
st.title("🏦 Loan Default Prediction App")
st.markdown("""
Enter the loan application details below to predict if the loan is likely to be **Fully Paid** or if there is a risk of it being **Charged Off**.
""")

# 3. Create Input Fields
# These match the columns_needed list in your notebook
st.sidebar.header("User Input Features")

def user_input_features():
    # Numeric Inputs
    loan_amnt = st.sidebar.number_input("Loan Amount ($)", min_value=500, max_value=40000, value=10000)
    term = st.sidebar.selectbox("Term (months)", options=[36, 60], index=0)
    int_rate = st.sidebar.slider("Interest Rate (%)", 5.0, 35.0, 12.0)
    installment = st.sidebar.number_input("Monthly Installment ($)", min_value=10.0, max_value=1600.0, value=300.0)
    annual_inc = st.sidebar.number_input("Annual Income ($)", min_value=1000, max_value=1000000, value=50000)
    dti = st.sidebar.slider("Debt-to-Income Ratio (DTI)", 0.0, 100.0, 15.0)
    emp_length = st.sidebar.slider("Employment Length (Years)", 0, 10, 5)

    # Categorical Inputs
    grade = st.sidebar.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    home_ownership = st.sidebar.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
    verification_status = st.sidebar.selectbox("Verification Status", ['Source Verified', 'Verified', 'Not Verified'])
    purpose = st.sidebar.selectbox("Purpose", [
        'debt_consolidation', 'credit_card', 'home_improvement', 'other', 
        'major_purchase', 'medical', 'small_business', 'car', 'vacation', 
        'moving', 'house', 'renewable_energy', 'wedding'
    ])

    data = {
        'loan_amnt': loan_amnt,
        'term': term,
        'int_rate': int_rate,
        'installment': installment,
        'grade': grade,
        'annual_inc': annual_inc,
        'dti': dti,
        'emp_length': emp_length,
        'home_ownership': home_ownership,
        'verification_status': verification_status,
        'purpose': purpose
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# 4. Display input summary
st.subheader("Loan Details Summary")
st.write(input_df)

# 5. Prediction Logic
if st.button("Predict Loan Status"):
    # The pipeline handles scaling and encoding internally if you saved the whole Pipeline object
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Prediction Result")
    
    # Target mapping from notebook: 0 = Fully Paid, 1 = Charged Off
    if prediction[0] == 0:
        st.success(f"✅ Prediction: **Fully Paid**")
        st.write(f"Confidence Level: {prediction_proba[0][0]:.2%}")
    else:
        st.error(f"⚠️ Prediction: **High Risk of Charge Off**")
        st.write(f"Risk Probability: {prediction_proba[0][1]:.2%}")

    # Display helpful stats
    st.info(f"Based on an interest rate of {input_df['int_rate'].values[0]}% and a DTI of {input_df['dti'].values[0]}.")
