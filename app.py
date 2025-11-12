import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

st.title("Loan Approval Prediction")

model = joblib.load("LoanValidater.pkl")

# Same 8 input features used during training
st.header("Enter Details")

person_age = st.number_input("Age", min_value=18, max_value=100)
person_income = st.number_input("Annual Income", min_value=0)
emp_exp = st.number_input("Employment Experience (Years)", min_value=0)
loan_amnt = st.number_input("Loan Amount", min_value=0)
loan_int_rate = st.number_input("Interest Rate", min_value=0.0)
cred_length = st.number_input("Credit History Length (Years)", min_value=0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900)

data = pd.DataFrame([[person_age, person_income, emp_exp, loan_amnt,
                      loan_int_rate, cred_length, credit_score]],
    columns=['person_age','person_income','person_emp_exp','loan_amnt',
             'loan_int_rate','cb_person_cred_hist_length','credit_score'])

# Apply scaling (must match training)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

if st.button("Predict"):
    pred = model.predict(scaled_data)[0]
    prob = model.predict_proba(scaled_data)[0][1] * 100

    if pred == 1:
        st.success(f"✅ Loan Approved (Probability: {prob:.2f}%)")
    else:
        st.error(f"❌ Loan Rejected (Approval Probability: {prob:.2f}%)")
