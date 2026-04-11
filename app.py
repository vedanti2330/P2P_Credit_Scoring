import streamlit as st
import pandas as pd
import pickle
import os

# 1. Page Configuration
st.set_page_config(page_title="Loan Default Prediction", page_icon="🏦")

# 2. Model Loading Logic
model_path = 'final_model.pkl'

@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

model = load_model()

# 3. UI Header
st.title("🏦 Loan Default Prediction App")
st.write("""
Enter the loan application details below to predict if the loan is likely to be **Fully Paid** or if there is a risk of it being **Charged Off**.
""")

# 4. Input Fields (Matching your screenshot)
st.subheader("Loan Details Summary")

col1, col2, col3 = st.columns(3)

with col1:
    loan_amnt = st.number_input("Loan Amount", value=10000)
    term = st.selectbox("Term (Months)", options=[36, 60], index=0)
    int_rate = st.number_input("Interest Rate (%)", value=12.0)

with col2:
    installment = st.number_input("Installment", value=300.0)
    grade = st.selectbox("Grade", options=['A', 'B', 'C', 'D', 'E', 'F', 'G'], index=0)
    annual_inc = st.number_input("Annual Income", value=50000)

with col3:
    dti = st.number_input("DTI", value=15.0)
    emp_length = st.number_input("Employment Length (Years)", value=5)
    home_ownership = st.selectbox("Home Ownership", options=['RENT', 'MORTGAGE', 'OWN', 'OTHER'], index=0)

# 5. Create the DataFrame (input_df)
input_data = {
    'loan_amnt': [loan_amnt],
    'term': [term],
    'int_rate': [int_rate],
    'installment': [installment],
    'grade': [grade],
    'annual_inc': [annual_inc],
    'dti': [dti],
    'emp_length': [emp_length],
    'home_ownership': [home_ownership]
}

input_df = pd.DataFrame(input_data)

# Display the table as seen in your screenshot
st.table(input_df)

# 6. Prediction Logic
if st.button("Predict Loan Status"):
    if model is not None:
        try:
            # Make prediction
            prediction = model.predict(input_df)
            
            # Display results
            st.subheader("Result:")
            if prediction[0] == 1: # Adjust based on your model's labels
                st.success("✅ Likely to be Fully Paid")
            else:
                st.error("⚠️ Risk of being Charged Off")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.info("Check if your input features match exactly what the model was trained on.")
    else:
        st.error(f"Model file '{model_path}' not found. Please upload it to your repository.")
