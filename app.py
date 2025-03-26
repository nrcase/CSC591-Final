import streamlit as st
import pandas as pd

st.title("BioTech Applied AI Final Project Application")

data_vis, predict, time_series, outliers = st.tabs(["Data Visualization", "ML Model Prediction", "Time Series Prediction for 30L", "Outliers Detection"])

with data_vis:
    df_choice = st.selectbox("Choose a dataset", ["1mL", "30L"])

    if df_choice == "1mL":
        uploaded_file = "data/1mL_Dataset.xlsx"
    elif df_choice == "30L":
        uploaded_file = "data/30L_Dataset.xlsx"

    df = pd.read_excel(uploaded_file)

    # Data preview
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Statistics summary
    st.subheader("Dataset Summary")
    st.write(df.describe())

    


