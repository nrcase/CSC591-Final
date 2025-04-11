import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import altair as alt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from model import ModelDevelopment

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

                # Get color from the color list
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
                                         text=[f"{avg_metric:.4f}"],
                                         textfont=dict(size=12, color="white"),
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

        if st.session_state.choice == "1mL":
            st.subheader("96-Well Plate Visualization")

            # Create a mapping for rows and columns to numerical indices
            row_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4,
                       'F': 5, 'G': 6, 'H': 7}  # Extend if you have more rows
            # Adjust range if you have more columns
            col_map = {str(i): i - 1 for i in range(1, 13)}

            # Create a new DataFrame with numerical row and column indices for the heatmap
            heatmap_df = df2.copy()
            # print(heatmap_df.columns)
            heatmap_df['Row_Index'] = heatmap_df['Row'].map(row_map)
            heatmap_df['Well_Index'] = heatmap_df['Well'].astype(
                str).map(col_map)

            # Drop rows with NaN in the index columns if your original data doesn't perfectly fill the plate
            heatmap_df = heatmap_df.dropna(
                subset=['Row_Index', 'Well_Index'])

            # Let the user choose which column to visualize
            color_column = st.selectbox("Select a column to visualize on the plate:",
                                        ['IPTG (mM)', 'Run Time (h)', 'Final OD (OD 600)',
                                         'GFPuv (g/L)', 'Total GFP (g)', 'Total Biomass (g)',
                                         'Biomass\n/Substrate\n(g/g)', 'Product\n/Substrate (g/g)',
                                         'Product\n/Biomass \n(g/g)', 'Volumetric productivity (g/hr*L)',
                                         'Growth Rate (1/h)'])

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

        elif st.session_state.choice == "30L":

            st.subheader("30L Bioreactor Visualization")

            # Get unique bioreactors
            bioreactors = sorted(df2['Bioreactor'].unique())
            n_bioreactors = len(bioreactors)

            # Define a layout
            n_cols = 3
            # Calculate number of rows needed
            n_rows = (n_bioreactors + n_cols - 1) // n_cols

            temp = df2.dropna(axis=1, how='all').drop(
                columns=['Bioreactor'])
            numerical_cols = temp.select_dtypes(
                include=['number']).columns.tolist()
            metric_to_visualize = st.selectbox(
                "Select metric to visualize:", numerical_cols)

            # Determine color scale
            min_val = df2[metric_to_visualize].min()
            max_val = df2[metric_to_visualize].max()
            color_scale_name = 'viridis'

            # Define a fallback viridis-like color list
            fallback_viridis = ["#440154", "#482878", "#3e4a89", "#31688e",
                                "#26828e", "#1f9e89", "#35b779", "#6ece58", "#b5dc36", "#fde725"]

            # Create subplots
            fig = make_subplots(rows=n_rows, cols=n_cols,
                                subplot_titles=[f"Bioreactor {br}" for br in bioreactors])

            row_idx, col_idx = 1, 1
            for i, bioreactor in enumerate(bioreactors):
                bioreactor_data = df2[df2['Bioreactor'] == bioreactor]
                avg_metric = bioreactor_data[metric_to_visualize].mean()

                # Normalize the metric value
                if max_val - min_val == 0:
                    normalized_value = 0
                else:
                    normalized_value = (
                        avg_metric - min_val) / (max_val - min_val)

                # Get color from the color list
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
                                         text=[f"{avg_metric:.4f}"],
                                         textfont=dict(size=12, color="white"),
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


def overall_tab(df):
    # Overall Data Preview
    st.subheader("Overall Data Preview")
    st.dataframe(df, use_container_width=True)

    # Statistics summary for overall dataset
    st.subheader("Overall Dataset Summary")
    # Focus on numeric columns for stats
    summary_df = calculate_summary(df)
    st.dataframe(summary_df, use_container_width=True)


def individual_tab(df):
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
        st.write(calculate_summary(selected_run_data.select_dtypes(
            include=['number'], exclude=['datetime'])))
        # st.write(selected_run_data.select_dtypes(
        #     include=['number'], exclude=['datetime']).describe())

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

def load_model_data():
    """Load and return the data pipeline"""
    try:
        # Use relative path from current directory
        project_root = "./"
        data_path = os.path.join(project_root, 'data', 'Combined_Dataset.xlsx')
        
        # Debug information
        st.write(f"Project root directory: {project_root}")
        st.write(f"Looking for data file at: {data_path}")
        st.write(f"File exists: {os.path.exists(data_path)}")
        
        if not os.path.exists(data_path):
            st.error(f"Data file not found at: {data_path}")
            # List files in the project root directory
            st.write("Files in project root directory:")
            for file in os.listdir(project_root):
                if file.endswith('.xlsx'):
                    st.write(f"- {file}")
            return None
        
        # Create the model development pipeline
        pipeline = ModelDevelopment(data_path)
        if pipeline.load_data():
            return pipeline
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Show stack trace for debugging
        import traceback
        st.code(traceback.format_exc())
        return None

def plot_actual_vs_predicted(y_test, y_pred, model_name, target):
    """Plot actual vs predicted values with interactive hover"""
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        name='Predictions',
        hovertemplate="<b>Actual:</b> %{x:.4f}<br><b>Predicted:</b> %{y:.4f}<extra></extra>"
    ))
    
    # Add perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f'{model_name}: Actual vs Predicted ({target})',
        xaxis_title='Actual',
        yaxis_title='Predicted',
        height=500,
        width=800,
        hovermode='closest',
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    return fig

def plot_pls_components(model, X, y, title):
    """Plot PLS components with interactive hover"""
    try:
        X_transformed = model.transform(X)
        
        # Create subplots
        fig = go.Figure()
        
        if model.n_components >= 2:
            # Plot first two components
            fig.add_trace(go.Scatter(
                x=X_transformed[:, 0],
                y=X_transformed[:, 1],
                mode='markers',
                marker=dict(
                    color=y,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Target Value')
                ),
                hovertemplate="<b>Component 1:</b> %{x:.4f}<br><b>Component 2:</b> %{y:.4f}<br><b>Target:</b> %{marker.color:.4f}<extra></extra>"
            ))
            fig.update_layout(
                xaxis_title='Component 1',
                yaxis_title='Component 2',
                title='First Two PLS Components'
            )
        else:
            # Plot single component against target
            fig.add_trace(go.Scatter(
                x=X_transformed[:, 0],
                y=y,
                mode='markers',
                hovertemplate="<b>Component 1:</b> %{x:.4f}<br><b>Target:</b> %{y:.4f}<extra></extra>"
            ))
            fig.update_layout(
                xaxis_title='Component 1',
                yaxis_title='Target Value',
                title='First PLS Component vs Target'
            )
        
        # Calculate explained variance
        var_explained = []
        total_var = np.var(X, axis=0).sum()
        X_transformed_full = model.transform(X)
        
        for i in range(model.n_components):
            X_transformed_i = np.zeros_like(X_transformed_full)
            X_transformed_i[:, :i+1] = X_transformed_full[:, :i+1]
            X_reconstructed_i = model.inverse_transform(X_transformed_i)
            unexplained_var = np.var(X - X_reconstructed_i, axis=0).sum()
            explained_var = (1 - unexplained_var / total_var) * 100
            var_explained.append(explained_var)
        
        # Create explained variance plot
        fig_var = go.Figure()
        fig_var.add_trace(go.Scatter(
            x=list(range(1, model.n_components + 1)),
            y=var_explained,
            mode='lines+markers',
            hovertemplate="<b>Components:</b> %{x}<br><b>Explained Variance:</b> %{y:.2f}%<extra></extra>"
        ))
        
        fig_var.update_layout(
            title='Explained Variance by Components',
            xaxis_title='Number of Components',
            yaxis_title='Cumulative Explained Variance (%)',
            height=400,
            width=800,
            hovermode='x unified',
            hoverlabel=dict(bgcolor="white", font_size=12)
        )
        
        return fig, fig_var
    except Exception as e:
        st.warning(f"Could not plot PLS components: {str(e)}")
        st.write(f"Debug info - X shape: {X.shape}, n_components: {model.n_components}")
        st.write(f"Model coefficients shape: {model.coef_.shape}")
        return None, None

def load_models_and_scalers(scale, target):
    """Load models, scaler, and label encoders for a given scale and target"""
    try:
        scale_clean = scale.replace(' ', '_')
        target_clean = target.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_per_')
        base_path = "./"
        
        # Load scaler and encoders first
        scaler = joblib.load(os.path.join(base_path, 'models', f"scaler_{scale_clean}_{target_clean}.joblib"))
        label_encoders = joblib.load(os.path.join(base_path, 'models', f"label_encoders_{scale_clean}_{target_clean}.joblib"))
        
        # Load models
        models = {}
        model_names = ['RandomForest', 'XGBoost', 'SVR', 'PLS']
        for model_name in model_names:
            model_path = os.path.join(base_path, 'models', f"{model_name}_{scale_clean}_{target_clean}.joblib")
            models[model_name] = joblib.load(model_path)
            
        return models, scaler, label_encoders
    except Exception as e:
        st.error(f"Error loading models and scalers: {str(e)}")
        return None, None, None


@st.cache_data
def load_data(url):
    df = pd.read_excel(url)
    return df


def load_data_sheet(url, sheet_name):
    df = pd.read_excel(url, sheet_name=sheet_name)
    return df


def calculate_summary(df):
    numeric_cols = df.select_dtypes(include="number").columns
    summary = {}
    for col in numeric_cols:
        summary[col] = {
            "Mean": df[col].mean(),
            "Min": df[col].min(),
            "Max": df[col].max(),
            "Mode": df[col].mode().iloc[0] if not df[col].mode().empty else None,
            "Standard Deviation": df[col].std(),
            "Q1": df[col].quantile(0.25),
            "Q3": df[col].quantile(0.75),
            "Median": df[col].median(),
            "Skewness": df[col].skew(),
            "Variance": df[col].var(),
            "Interquartile Range": df[col].quantile(0.75) - df[col].quantile(0.25),
            "Range": df[col].max() - df[col].min(),
            "Monotonicity": (
                "Increasing" if df[col].is_monotonic_increasing
                else "Decreasing" if df[col].is_monotonic_decreasing
                else "Non-Monotonic"
            ),
            "Kurtosis": df[col].kurt(),
            "Coefficient of Variation": df[col].std() / df[col].mean() if df[col].mean() != 0 else None,
            "95th Percentile": df[col].quantile(0.95),
            "5th Percentile": df[col].quantile(0.05),
        }
    return pd.DataFrame(summary).T


def main():
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

    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = load_model_data()

    if st.session_state.pipeline is None:
        st.error("Failed to load data. Please check the data file and try again.")
        return

    # Title and description
    st.title("BioInsight Application")

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
        st.header("Welcome to BioInsight!")
        st.markdown("""
        Welcome to the BioInsight Application! This app allows you to explore datasets, visualize data, 
        and perform advanced analytics such as machine learning predictions and time series analysis.
        Use the navigation bar on the left to select a feature.
        
        Navigate through the sidebar to explore different features of this application:
        - **Data Visualization** allows you to explore datasets interactively.
        - **KPI and Initial Conditions Visualization** provides insights into key performance indicators and initial conditions.
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
                overall_tab(df)

            with individual:
                individual_tab(df)

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

    elif page == "ML Model Prediction":
        st.header("Model Results and Comparison")
        
        # Scale and target selection
        scale = st.selectbox("Select Scale", ["1 mL", "30 L"])
        target = st.selectbox("Select Target", 
            ["Final OD (OD 600)", "GFPuv (g/L)"])
        
        try:
            # Prepare data
            X, y, feature_cols = st.session_state.pipeline.prepare_data(scale, target)
            X_selected, _ = st.session_state.pipeline.select_features(X, y)
            
            # Load models and scalers
            models, scaler, label_encoders = load_models_and_scalers(scale, target)
            
            if models is None:
                st.error("Failed to load models. Please ensure models have been trained.")
                return
            
            # Evaluate each model
            for model_name, model in models.items():
                try:
                    # Make predictions
                    if model_name == 'PLS':
                        # For PLS, we use all features since it handles feature selection internally
                        y_pred = model.predict(X_selected)
                    else:
                        y_pred = model.predict(X_selected)
                    
                    rmse = np.sqrt(mean_squared_error(y, y_pred))
                    r2 = r2_score(y, y_pred)
                    
                    # Display results
                    st.subheader(f"{model_name} Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Model Performance:")
                        st.write(f"RMSE: {rmse:.4f}")
                        st.write(f"R2 Score: {r2:.4f}")
                        
                        # Show feature importance if available
                        if hasattr(model, 'feature_importances_'):
                            importance = pd.DataFrame({
                                'feature': X_selected.columns,
                                'importance': model.feature_importances_
                            }).sort_values('importance', ascending=False)
                            st.write("\nTop 5 Important Features:")
                            st.write(importance.head())
                        
                        # Show PLS-specific information
                        if model_name == 'PLS':
                            st.write("\nPLS Model Information:")
                            st.write(f"Number of components: {model.n_components}")
                            
                            # Calculate explained variance
                            X_transformed = model.transform(X_selected)
                            X_reconstructed = model.inverse_transform(X_transformed)
                            total_var = np.var(X_selected, axis=0).sum()
                            explained_var = 1 - np.var(X_selected - X_reconstructed, axis=0).sum() / total_var
                            st.write(f"Total explained variance: {explained_var * 100:.2f}%")
                            
                            # Show loadings
                            loadings = pd.DataFrame(
                                model.x_loadings_,
                                columns=[f'Component {i+1}' for i in range(model.n_components)],
                                index=X_selected.columns
                            )
                            st.write("\nComponent Loadings (top 5 features):")
                            st.write(loadings.abs().sum(axis=1).sort_values(ascending=False).head())
                    
                    with col2:
                        st.plotly_chart(plot_actual_vs_predicted(y, y_pred, model_name, target),
                            use_container_width=True)
                    
                    # Show PLS components plot
                    if model_name == 'PLS':
                        fig_components, fig_variance = plot_pls_components(model, X_selected, y, 
                            f"PLS Components Analysis - {scale}, {target}")
                        if fig_components is not None:
                            st.plotly_chart(fig_components, use_container_width=True)
                            st.plotly_chart(fig_variance, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"Could not evaluate {model_name} model: {str(e)}")
            
            # Interactive prediction
            if models:
                st.subheader("Interactive Prediction")
                st.write("Select feature values for prediction:")
                
                input_features = {}
                for feature in X_selected.columns:
                    mean_val = float(X_selected[feature].mean())
                    std_val = float(X_selected[feature].std())
                    input_features[feature] = st.slider(
                        feature,
                        min_value=mean_val - 2*std_val,
                        max_value=mean_val + 2*std_val,
                        value=mean_val,
                        format="%.2f"
                    )
                
                if st.button("Predict"):
                    input_df = pd.DataFrame([input_features])
                    st.write("\nPredictions:")
                    for model_name, model in models.items():
                        try:
                            # All models can use the same input features since PLS handles feature selection internally
                            prediction = model.predict(input_df)[0]
                            st.write(f"{model_name}: {prediction:.4f}")
                        except Exception as e:
                            st.warning(f"Could not make prediction with {model_name}: {str(e)}")
            
        except Exception as e:
            st.error(f"Error in model evaluation: {str(e)}")
            st.write("Please ensure models have been trained and saved correctly.")

if __name__ == "__main__":
    main()
