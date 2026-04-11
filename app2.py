import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Page configuration
st.set_page_config(page_title="P2P Credit Scoring Dashboard", layout="wide")

st.title("📊 P2P Credit Scoring Prediction")
st.markdown("""
This app predicts the likelihood of loan default based on historical P2P lending data.
""")

# 1. Sidebar for Data Upload and Model Selection
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload your CSV (e.g., accepted_2007_to_2018Q4.csv)", type=["csv"])

model_option = st.sidebar.selectbox(
    "Select Model",
    ("Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost")
)

# 2. Data Loading and Cleaning Function
@st.cache_data
def load_and_clean_data(file):
    # Load sample for performance (following notebook logic)
    df = pd.read_csv(file, low_memory=False, nrows=20000)
    
    # Feature Selection
    columns_needed = [
        'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 
        'annual_inc', 'dti', 'emp_length', 'home_ownership', 
        'verification_status', 'purpose', 'loan_status'
    ]
    df = df[columns_needed].dropna(subset=['loan_status'])
    
    # Filter for completed loans
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
    
    # Target Encoding
    df['target'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})
    df.drop('loan_status', axis=1, inplace=True)
    
    # Feature Engineering
    df['term'] = df['term'].str.replace(' months', '', regex=False).astype(float)
    df['int_rate'] = df['int_rate'].astype(str).str.replace('%', '', regex=False).astype(float)
    df['emp_length'] = df['emp_length'].astype(str).str.extract('(\d+)').astype(float)
    
    # Handle Missing Values
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    # Categorical Encoding for modeling
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    return df, df_encoded

if uploaded_file is not None:
    raw_df, model_df = load_and_clean_data(uploaded_file)
    
    # 3. Display Data Overview
    st.subheader("Data Preview")
    st.write(raw_df.head())
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Data Shape:**", raw_df.shape)
    with col2:
        st.write("**Target Distribution:**")
        st.bar_chart(raw_df['target'].value_counts())

    # 4. Model Training Logic
    st.divider()
    st.subheader(f"Training {model_option} Model")
    
    X = model_df.drop('target', axis=1)
    y = model_df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Dictionary to map selection to Sklearn/XGB objects
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "XGBoost": XGBClassifier()
    }

    model = models[model_option]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 5. Results and Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    m_col1, m_col2 = st.columns(2)
    m_col1.metric("Accuracy", f"{acc:.2%}")
    m_col2.metric("ROC-AUC Score", f"{auc:.2f}")

    st.text("Classification Report:")
    st.code(classification_report(y_test, y_pred))

    # 6. Visualization: Confusion Matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)

else:
    st.info("Please upload a CSV file to get started.")
