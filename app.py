import streamlit as st
import pandas as pd
import numpy as np


@st.cache_data
def load_data(url):
    df = pd.read_excel(url)
    return df


# Set page configuration
st.set_page_config(page_title="BioTech AI Application",
                   layout="wide", page_icon=":material/science:")

# Title and description
st.title("BioTech Applied AI Application")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a feature:",
    ["Home", "Data Visualization", "ML Model Prediction",
        "Time Series Prediction for 30L", "Outliers Detection"]
)

# Home Page
if page == "Home":
    st.header("Welcome to BioTech AI Application!")
    st.markdown("""
    Welcome to the BioTech AI Application! This app allows you to explore datasets, visualize data, 
    and perform advanced analytics such as machine learning predictions and time series analysis.
    Use the navigation bar on the left to select a feature.
    
    Navigate through the sidebar to explore different features of this application:
    - **Data Visualization** allows you to explore datasets interactively.
    - **ML Model Prediction** enables you to predict outcomes using machine learning models.
    - **Time Series Prediction for 30L** provides time series forecasting for 30L dataset. This feature is helpful for when 30L batches take a long time,
    so when taking samples at predictable intervals, we can predict the future values.
    - **Outliers Detection** helps you identify anomalies in within each dataset.
    """)

# Data Visualization Page
elif page == "Data Visualization":
    st.header("Data Visualization")

    # Dataset selection
    st.markdown("### Select a Dataset")
    df_choice = st.selectbox(
        "Choose a dataset:",
        ["1mL", "30L"],
        help="Select the dataset you want to visualize.",
        placeholder="Select a dataset",
        index=None,
    )

    # Load the selected dataset
    if df_choice == "1mL":
        uploaded_file = "data/1mL_Dataset.xlsx"
    elif df_choice == "30L":
        uploaded_file = "data/30L_Dataset.xlsx"
    else:
        st.stop()

    try:
        df = load_data(uploaded_file)
        st.success(f"Loaded {df_choice} dataset successfully!")

        # Overall Data Preview
        st.markdown("---")
        st.subheader("Overall Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        # Statistics summary for overall dataset
        st.subheader("Overall Dataset Summary")
        # Focus on numeric columns for stats
        df_numeric = df.select_dtypes(include=['number'], exclude=['datetime'])
        st.write(df_numeric.describe())

        # Individual Run Data Visualization
        st.markdown("---")
        st.subheader("Individual Run Data Visualization")

        run_names = df["Batch"].unique()
        run_choice = st.selectbox(
            "Choose a run:",
            run_names,
            help="Select a specific batch/run to visualize its data.",
            index=None,
            placeholder="Select a run",
        )

        if run_choice:
            selected_run_data = df[df["Batch"] == run_choice]
            st.write(selected_run_data.head())

            # Individual Run Summary Statistics
            st.subheader(f"Summary Statistics for Batch: {run_choice}")
            # df_numeric = df.select_dtypes(include=['number'], exclude=['datetime'])
            st.write(selected_run_data.select_dtypes(
                include=['number'], exclude=['datetime']).describe())

            if st.checkbox("Show charts"):
                cant_use = [
                    'Well', 'Culture Time (h)', 'IPTG (mM)', 'Bioreactor']
                numeric_columns = selected_run_data.select_dtypes(
                    include=np.number).columns
                filtered_columns = [
                    col for col in numeric_columns if col not in cant_use]

                y_choice = st.selectbox("Select target variable:",
                                        filtered_columns,
                                        help="Select a target variable to visualize.",
                                        index=None,
                                        placeholder="Select a target variable")

                st.scatter_chart(selected_run_data.select_dtypes(
                    include=['number']), x='Culture Time (h)', y=y_choice)

                st.bar_chart(selected_run_data.select_dtypes(
                    include=['number']), x='Culture Time (h)', y=y_choice)

        else:
            st.stop()

    except Exception as e:
        st.error(f"Error loading dataset: {e}")
