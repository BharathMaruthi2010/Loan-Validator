# Loan-Validator
This project uses a Logistic Regression model to predict whether a loan should be approved based on customer financial details. The data is scaled using StandardScaler, and only the numerical features used during training are taken as input in the Streamlit UI.

#preview
<img width="1171" height="598" alt="Screenshot 2025-11-12 144046" src="https://github.com/user-attachments/assets/f1ef3cd7-599a-4380-9627-1cb411989788" />

<img width="1110" height="541" alt="Screenshot 2025-11-12 144102" src="https://github.com/user-attachments/assets/781e6396-d46c-4f22-af6a-a69e77691136" />


Loan Approval Prediction using Logistic Regression & Streamlit

This project builds a Loan Approval Prediction System using a machine learning model trained with Logistic Regression.
The model uses customer financial details to decide whether a loan should be approved. A Streamlit web interface is provided to make real-time predictions.

✅ How it Works

The dataset (loan_data.csv) is loaded and preprocessed

Numerical features are scaled using StandardScaler

The data is split into training and testing sets

A Logistic Regression model is trained

The trained model is saved as LoanValidater.pkl

Streamlit UI takes user inputs and predicts:
✔ Loan Approved / Rejected
✔ Approval probability

📊 Input Features Used

The prediction is made using these numerical features:

Age

Annual Income

Employment Experience

Loan Amount

Interest Rate

Credit History Length

Credit Score

🚀 Run the Streamlit App
pip install streamlit pandas numpy scikit-learn joblib
streamlit run app.py

📁 Project Files
File	Purpose
loan_data.csv	Original dataset
LoanValidater.pkl	Trained Logistic Regression model
app.py	Streamlit user interface

This project demonstrates how machine learning can be integrated with a simple UI for real-world loan approval decisions.
