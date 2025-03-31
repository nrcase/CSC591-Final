import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.colors as colors
import altair as alt


def IC_tab():
    st.header("Initial Conditions Visualization")

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
        df = load_data_sheet(uploaded_file, "Initial Conditions")
        st.success(f"Loaded {df_choice} dataset successfully!")

        st.markdown("---")
        st.subheader("Initial Conditions Data Preview")
        st.dataframe(df, use_container_width=True)

        if df_choice == "1mL":
            st.subheader("96-Well Plate Visualization")

            # Create a mapping for rows and columns to numerical indices
            row_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4,
                       'F': 5, 'G': 6, 'H': 7}  # Extend if you have more rows
            # Adjust range if you have more columns
            col_map = {str(i): i - 1 for i in range(1, 13)}

            # Create a new DataFrame with numerical row and column indices for the heatmap
            heatmap_df = df.copy()
            heatmap_df['Row_Index'] = heatmap_df['Row'].map(row_map)
            heatmap_df['Well_Index'] = heatmap_df['Well'].astype(
                str).map(col_map)

            # Drop rows with NaN in the index columns if your original data doesn't perfectly fill the plate
            heatmap_df = heatmap_df.dropna(
                subset=['Row_Index', 'Well_Index'])

            # Let the user choose which column to visualize
            color_column = st.selectbox("Select a column to visualize on the plate:",
                                        ['Inoculation OD', 'IPTG (mM)', 'Initial Glucose (g/L)'])

            # Create the heatmap
            fig = px.imshow(heatmap_df.pivot_table(index='Row_Index', columns='Well_Index', values=color_column),
                            labels=dict(x="Well", y="Row",
                                        color=color_column),
                            # Adjust range based on your 'Well' numbers
                            x=[str(i) for i in range(1, 9)],
                            # Adjust range based on your 'Row' letters
                            y=list(row_map.keys())[:6],
                            color_continuous_scale="viridis",
                            title=f"Plate View of {color_column}")

            # Customize layout for better readability
            fig.update_layout(xaxis_side="top",
                              margin=dict(t=150, b=20, l=20, r=20))  # Adjust top margin (t)

            # Add tooltips to show all the information on hover
            fig.update_traces(
                hovertemplate="<b>Row:</b> %{y}<br><b>Well:</b> %{x}<br><b>Value</b>: %{z}<extra></extra>")

            st.plotly_chart(fig)

        elif df_choice == "30L":

            st.subheader("30L Bioreactor Visualization")

            # Get unique bioreactors
            bioreactors = sorted(df['Bioreactor'].unique())
            n_bioreactors = len(bioreactors)

            # Define a layout
            n_cols = 3
            # Calculate number of rows needed
            n_rows = (n_bioreactors + n_cols - 1) // n_cols

            temp = df.dropna(axis=1, how='all').drop(
                columns=['Bioreactor'])
            numerical_cols = temp.select_dtypes(
                include=['number']).columns.tolist()
            metric_to_visualize = st.selectbox(
                "Select metric to visualize:", numerical_cols)

            # Determine color scale
            min_val = df[metric_to_visualize].min()
            max_val = df[metric_to_visualize].max()
            color_scale_name = 'viridis'

            # Define a fallback viridis-like color list
            fallback_viridis = ["#440154", "#482878", "#3e4a89", "#31688e",
                                "#26828e", "#1f9e89", "#35b779", "#6ece58", "#b5dc36", "#fde725"]

            # Create subplots
            fig = make_subplots(rows=n_rows, cols=n_cols,
                                subplot_titles=[f"Bioreactor {br}" for br in bioreactors])

            row_idx, col_idx = 1, 1
            for i, bioreactor in enumerate(bioreactors):
                bioreactor_data = df[df['Bioreactor'] == bioreactor]
                avg_metric = bioreactor_data[metric_to_visualize].mean()

                # Normalize the metric value
                if max_val - min_val == 0:
                    normalized_value = 0
                else:
                    normalized_value = (
                        avg_metric - min_val) / (max_val - min_val)

                # Get color from the fallback list
                color_index = int(normalized_value *
                                  (len(fallback_viridis) - 1))
                color_val = fallback_viridis[color_index]

                # Add colored background
                fig.add_trace(go.Scatter(x=[0.5], y=[0.5],
                                         mode='markers',
                                         marker=dict(
                                             size=50, color=color_val),
                                         showlegend=False),
                              row=row_idx, col=col_idx)
                # Add text annotation
                fig.add_trace(go.Scatter(x=[0.5], y=[0.5],
                                         mode='text',
                                         text=[f"{avg_metric:.2f}"],
                                         textfont_size=12,
                                         showlegend=False),
                              row=row_idx, col=col_idx)

                # Customize subplot layout
                fig.update_xaxes(visible=False, range=[
                    0, 1], row=row_idx, col=col_idx)
                fig.update_yaxes(visible=False, range=[
                    0, 1], row=row_idx, col=col_idx)

                col_idx += 1
                if col_idx > n_cols:
                    col_idx = 1
                    row_idx += 1

            fig.update_layout(
                title_text=f"Average {metric_to_visualize} by Bioreactor (Darker = Less)")
            st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error loading dataset: {e}")


def KPI_tab():
    st.header("KPI Visualization")
    # Dataset selection
    st.markdown("### Select a Dataset")
    st.session_state.choice = st.selectbox(
        "Choose a dataset:",
        ["1mL", "30L"],
        help="Select the dataset you want to visualize.",
        placeholder="Select a dataset to visualize",
        index=None,
    )

    # Load the selected dataset
    if st.session_state.choice == "1mL":
        file = "data/1mL_Dataset.xlsx"
    elif st.session_state.choice == "30L":
        file = "data/30L_Dataset.xlsx"
    else:
        st.stop()

    try:
        df2 = load_data_sheet(file, "Process KPIs")
        st.success(
            f"Loaded {st.session_state.choice} dataset successfully!")

        st.markdown("---")
        st.subheader("Process KPI Data Preview")
        st.dataframe(df2, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading dataset: {e}")


def overall_tab():
    # Overall Data Preview
    st.subheader("Overall Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Statistics summary for overall dataset
    st.subheader("Overall Dataset Summary")
    # Focus on numeric columns for stats
    df_numeric = df.select_dtypes(
        include=['number'], exclude=['datetime'])
    st.write(df_numeric.describe())


def individual_tab():
    # Individual Run Data Visualization
    st.subheader("Individual Run Data Visualization")

    run_names = df["Batch"].unique()
    run_choices = st.multiselect(
        "Choose one or more runs:",
        options=run_names,
        help="Select one or more runs to visualize their data.",
        placeholder="Select run(s)",
    )

    if run_choices:
        # Filter the DataFrame for the selected run
        selected_run_data = df[df["Batch"].isin(run_choices)]
        st.write(selected_run_data.head())

        # Individual Run Summary Statistics
        st.subheader(f"Summary Statistics for Batch: {run_choices}")
        st.write(selected_run_data.select_dtypes(
            include=['number'], exclude=['datetime']).describe())

        # Optional charts
        if st.checkbox("Show charts"):
            # Columns to exclude from visualization
            cant_use = [
                'Well', 'Culture Time (h)', 'IPTG (mM)', 'Bioreactor']
            numeric_columns = selected_run_data.select_dtypes(
                include=np.number).columns
            filtered_columns = [
                col for col in numeric_columns if col not in cant_use]

            if filtered_columns:
                # Select a target variable for visualization
                y_choice = st.selectbox("Select target variable:",
                                        filtered_columns,
                                        help="Select a target variable to visualize.",
                                        index=None,
                                        placeholder="Select a target variable")

                scatter_chart = alt.Chart(selected_run_data).mark_circle(size=100).encode(
                    x="Culture Time (h):Q",
                    y=f"{y_choice}:Q",
                    color=alt.Color("Batch:N", legend=alt.Legend(
                        title="Runs"))  # Distinct color for each run
                ).properties(
                    title="Scatter Chart"
                ).interactive()
                st.altair_chart(
                    scatter_chart, use_container_width=True)

                # Bar Chart with Altair (distinct colors per run)
                bar_chart = alt.Chart(selected_run_data).mark_bar().encode(
                    x="Culture Time (h):O",
                    y=f"{y_choice}:Q",
                    color=alt.Color("Batch:N", legend=alt.Legend(
                        title="Runs"))  # Distinct color for each run
                ).properties(
                    title="Bar Chart"
                ).interactive()
                st.altair_chart(bar_chart, use_container_width=True)

        else:
            st.stop()


@st.cache_data
def load_data(url):
    df = pd.read_excel(url)
    return df


def load_data_sheet(url, sheet_name):
    df = pd.read_excel(url, sheet_name=sheet_name)
    return df


# Set page configuration
st.set_page_config(page_title="BioTech AI Application",
                   layout="wide", page_icon=":material/science:")

st.markdown("""
<style>
* {
    overflow-anchor: none !important;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("BioTech Applied AI Application")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a feature:",
    ["Home", "Data Visualization", "KPI and Initial Conditions Visualization", "ML Model Prediction",
        "Time Series Prediction for 30L", "Outliers Detection"]
)

# Home Page
if page == "Home":
    st.cache_data.clear()
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
    st.cache_data.clear()

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
        overall, individual = st.tabs(
            ["Overall Data Visualization", "Individual Run Data Visualization"])
        with overall:
            overall_tab()

        with individual:
            individual_tab()

    except Exception as e:
        st.error(f"Error loading dataset: {e}")

elif page == "KPI and Initial Conditions Visualization":
    st.cache_data.clear()
    KPI, IC = st.tabs(
        ["KPI Visualization", "Inital Conditions Visualization"])

    with KPI:
        KPI_tab()

    with IC:
        IC_tab()
