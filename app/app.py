import pandas as pd
import streamlit as st

from src.predict import predict_dataframe

st.set_page_config(page_title="Bank Loan Prediction", layout="centered")
st.title("Bank Loan Prediction Demo")
st.caption("Deployment-style demo for the final notebook-selected Logistic Regression model.")

uploaded = st.file_uploader("Upload a CSV with raw input columns", type=["csv"])

st.markdown("### Expected raw columns")
st.code(
    "Age, Experience, Income, ZIP Code, Family, CCAvg, Education, Mortgage, Securities Account, CD Account, Online, CreditCard"
)

if uploaded is not None:
    data = pd.read_csv(uploaded)
    preds = predict_dataframe(data)
    st.success("Predictions generated.")
    st.dataframe(preds)
else:
    st.info("Upload a CSV file after training the model with `py -m scripts.run_training`.")
