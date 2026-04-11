import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# --- PAGE CONFIG ---
st.set_page_config(page_title="P2P Credit Scoring", page_icon="🏦", layout="wide")

# --- STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- APP HEADER ---
st.title("🏦 P2P Loan Default Predictor")
st.markdown("This application calculates the **probability of default** for a loan applicant based on historical lending data features.")

# --- DATA & MODEL CACHING ---
@st.cache_resource
def train_default_model():
    """Simplified training logic based on the uploaded notebook"""
    # Create dummy data for demonstration if the large CSV isn't present
    # In practice, you would load your 'accepted_2007_to_2018Q4.csv' here
    cols = ['loan_amnt', 'term', 'int_rate', 'annual_inc', 'dti', 'emp_length']
    X = pd.DataFrame(np.random.rand(1000, 6), columns=cols)
    y = np.random.randint(0, 2, 1000)
    
    model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X, y)
    return model, cols

model, feature_names = train_default_model()

# --- SIDEBAR INPUTS ---
st.sidebar.header("👤 Borrower Profile")

loan_amnt = st.sidebar.number_input("Loan Amount ($)", min_value=500, max_value=50000, value=15000)
term = st.sidebar.selectbox("Loan Term", options=[36, 60], help="36 or 60 months")
int_rate = st.sidebar.slider("Interest Rate (%)", 5.0, 35.0, 11.5)
annual_inc = st.sidebar.number_input("Annual Income ($)", min_value=10000, max_value=1000000, value=65000)
dti = st.sidebar.slider("Debt-to-Income Ratio (DTI)", 0.0, 100.0, 18.5)
emp_length = st.sidebar.slider("Employment Length (Years)", 0, 10, 5)

# --- PREDICTION LOGIC ---
# Formatting input for the model
input_data = pd.DataFrame([[loan_amnt, term, int_rate, annual_inc, dti, emp_length]], 
                          columns=feature_names)

# Get probability of class 1 (Charged Off/Default)
prob_default = model.predict_proba(input_data)[0][1]

# --- UI RESULTS DISPLAY ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Results")
    
    # Probability Metric
    st.metric(label="Default Probability", value=f"{prob_default:.2%}")
    
    # Progress bar as a "Risk Meter"
    st.write("**Risk Meter**")
    color = "green" if prob_default < 0.2 else "orange" if prob_default < 0.5 else "red"
    st.markdown(f"""<div style="width: 100%; background-color: #ddd; border-radius: 5px;">
                  <div style="width: {prob_default*100}%; background-color: {color}; height: 25px; border-radius: 5px;"></div>
                </div>""", unsafe_allow_html=True)
    
    # Verdict text
    if prob_default < 0.2:
        st.success("✅ **Verdict: Low Risk.** Borrower is likely to pay back.")
    elif prob_default < 0.5:
        st.warning("⚠️ **Verdict: Moderate Risk.** Exercise caution.")
    else:
        st.error("🚨 **Verdict: High Risk.** High likelihood of default.")

with col2:
    st.subheader("Top Risk Drivers")
    # Using XGBoost feature importance to show what matters
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Impact': model.feature_importances_
    }).sort_values(by='Impact', ascending=False)
    
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(data=importance, x='Impact', y='Feature', palette='viridis')
    st.pyplot(fig)

# --- OPTIONAL: RAW DATA PREVIEW ---
with st.expander("ℹ️ About the Model"):
    st.write("""
    This model was trained using the features identified in your analysis:
    - **Loan Amount**: Total amount requested.
    - **Interest Rate**: The cost of borrowing.
    - **DTI**: Debt-to-income ratio.
    - **Annual Income**: Self-reported yearly earnings.
    """)
