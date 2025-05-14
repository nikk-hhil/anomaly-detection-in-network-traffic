import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import time
from frontend.utils.visualization import prepare_dataframe_for_plotting

# Add the project root to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from src modules
from src.data_loader import load_dataset, get_dataset_info, get_column_statistics
from src.preprocessor import DataPreprocessor

def show_data_page():
    """Display the data upload and exploration page."""
    st.title("📊 Data Upload & Exploration")
    
    # Create tabs for the data workflow
    tabs = st.tabs(["Upload Data", "Data Overview", "Feature Analysis", "Preprocessing", "Feature Engineering"])
    
    # Tab 1: Data Upload
    with tabs[0]:
        show_data_upload_tab()
    
    # Only show other tabs if data is loaded
    if st.session_state.data is not None:
        # Tab 2: Data Overview
        with tabs[1]:
            show_data_overview_tab(st.session_state.data)
        
        # Tab 3: Feature Analysis
        with tabs[2]:
            show_feature_analysis_tab(st.session_state.data)
        
        # Tab 4: Preprocessing
        with tabs[3]:
            show_preprocessing_tab(st.session_state.data)
        
        # Tab 5: Feature Engineering
        with tabs[4]:
            show_feature_engineering_tab()
    else:
        # Display placeholder content if no data is loaded
        for i in range(1, 5):
            with tabs[i]:
                st.info("Please upload data first.")

def show_data_upload_tab():
    """Show data upload functionality."""
    st.header("Upload Network Traffic Data")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file with network traffic data", type="csv")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sample dataset option
        st.markdown("### Or use a sample dataset")
        use_sample = st.button("Load Sample Dataset")
        
        if use_sample:
            try:
                # Check if sample data exists
                sample_path = "data/test_data.csv"
                if os.path.exists(sample_path):
                    with st.spinner("Loading sample dataset..."):
                        data = pd.read_csv(sample_path)
                        st.session_state.data = data
                        st.session_state.target_column = ' Label' if ' Label' in data.columns else None
                        st.success(f"Loaded sample dataset with {data.shape[0]} rows and {data.shape[1]} columns!")
                else:
                    st.error("Sample dataset not found. Please upload your own data.")
            except Exception as e:
                st.error(f"Error loading sample dataset: {str(e)}")
    
    with col2:
        # Dataset information
        st.markdown("### Dataset Requirements")
        st.markdown("""
        The dataset should:
        - Be in CSV format
        - Contain network flow features
        - Ideally have a target column with attack labels
        
        Example features:
        - Flow duration
        - Packet counts
        - Byte counts
        - Protocol information
        """)
    
    # Process the uploaded file
    if uploaded_file is not None:
        try:
            with st.spinner("Loading and analyzing data..."):
                # Load the data
                data = pd.read_csv(uploaded_file)
                
                for col in data.columns:
                    if str(data[col].dtype).startswith(('Int', 'Float')):
                        data[col] = data[col].astype(float)
                
                # Store in session state
                st.session_state.data = data
                
                # Reset other session state variables that depend on the data
                st.session_state.preprocessed_data = None
                st.session_state.features = None
                st.session_state.trained_models = {}
                st.session_state.best_model = None
                st.session_state.evaluation_results = None
                st.session_state.predictions = None
                
                # Try to identify target column
                potential_target = identify_target_column(data)
                st.session_state.target_column = potential_target
                
                # Success message
                st.success(f"Successfully loaded data with {data.shape[0]} rows and {data.shape[1]} columns!")
                
                # Show quick preview
                st.subheader("Data Preview")
                st.dataframe(data.head())
        
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

def identify_target_column(data):
    """Try to identify the target column in the data."""
    # Look for common target column names
    target_keywords = ['label', 'class', 'target', 'attack', 'category']
    
    for col in data.columns:
        # Check if any keywords appear in the column name (case-insensitive)
        if any(keyword in col.lower() for keyword in target_keywords):
            return col
        
        # Also check if the column has limited unique values (typical for target columns)
        unique_count = data[col].nunique()
        if unique_count > 1 and unique_count <= 15 and data[col].dtype == 'object':
            return col
    
    return None

def show_data_overview_tab(data):
    """Show data overview including statistics and class distribution."""
    st.header("Dataset Overview")
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{data.shape[0]:,}")
    col2.metric("Columns", f"{data.shape[1]:,}")
    col3.metric("Missing Values", f"{data.isnull().sum().sum():,}")
    col4.metric("Duplicates", f"{data.duplicated().sum():,}")
    
    # Column types
    st.subheader("Column Data Types")
    dtype_counts = data.dtypes.value_counts().reset_index()
    dtype_counts.columns = ['Data Type', 'Count']
    
    # Prepare for plotting
    dtype_counts = prepare_dataframe_for_plotting(dtype_counts)
    
    fig = px.bar(
        dtype_counts, 
        x='Data Type', 
        y='Count',
        color='Data Type',
        text='Count'
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Target column selection
    st.subheader("Target Column Selection")
    
    target_columns = []
    for col in data.columns:
        if data[col].nunique() <= 20:  # Only consider columns with reasonable number of classes
            target_columns.append(col)
    
    # Default to the automatically identified target column
    default_idx = 0
    if st.session_state.target_column in target_columns:
        default_idx = target_columns.index(st.session_state.target_column)
    
    selected_target = st.selectbox(
        "Select the target column (with attack labels)",
        options=target_columns,
        index=default_idx,
        key="target_column_select"
    )
    
    if st.button("Set as Target Column"):
        st.session_state.target_column = selected_target
        st.success(f"Set '{selected_target}' as the target column!")
    
    # Show class distribution if target column is selected
    if st.session_state.target_column:
        st.subheader(f"Class Distribution for '{st.session_state.target_column}'")
        
        # Get class counts
        target_col = st.session_state.target_column
        class_counts = data[target_col].value_counts().reset_index()
        class_counts.columns = ['Class', 'Count']
        class_counts['Percentage'] = 100 * class_counts['Count'] / class_counts['Count'].sum()
        
        # Create visualization
        fig = px.bar(
            class_counts,
            x='Class',
            y='Count',
            color='Class',
            text=class_counts['Percentage'].apply(lambda x: f'{x:.1f}%'),
            title=f"Distribution of {target_col}"
        )
        fig.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed class information
        st.dataframe(class_counts, use_container_width=True)

def show_feature_analysis_tab(data):
    """Show feature analysis including correlations and distributions."""
    st.header("Feature Analysis")
    
    # Create two columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Feature selection for analysis
        st.subheader("Select Features")
        
        # Get numerical columns for analysis
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if not numeric_cols:
            st.warning("No numerical features found for analysis.")
            return
        
        # Select features to analyze
        selected_features = st.multiselect(
            "Select features to analyze",
            options=numeric_cols,
            default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
        )
        
        if not selected_features:
            st.info("Please select at least one feature to analyze.")
            return
        
        # Select visualization type
        viz_type = st.radio(
            "Select visualization type",
            ["Correlation Heatmap", "Feature Distributions", "Box Plots", "Scatter Plot"]
        )
    
    with col2:
        # Display selected visualization
        if viz_type == "Correlation Heatmap":
            show_correlation_heatmap(data, selected_features)
        elif viz_type == "Feature Distributions":
            show_feature_distributions(data, selected_features)
        elif viz_type == "Box Plots":
            show_box_plots(data, selected_features)
        elif viz_type == "Scatter Plot":
            show_scatter_plot(data, selected_features)

def show_correlation_heatmap(data, features):
    """Show correlation heatmap for selected features."""
    st.subheader("Correlation Heatmap")
    
    if len(features) < 2:
        st.warning("Please select at least 2 features for correlation analysis.")
        return
    
    # Calculate correlation matrix
    corr_matrix = data[features].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        aspect="auto"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Identify highly correlated features
    threshold = 0.8
    high_corr = []
    
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr.append((features[i], features[j], corr_matrix.iloc[i, j]))
    
    if high_corr:
        st.subheader("Highly Correlated Features")
        for feat1, feat2, corr in high_corr:
            st.write(f"**{feat1}** and **{feat2}**: {corr:.2f}")

def show_feature_distributions(data, features):
    """Show distribution plots for selected features."""
    st.subheader("Feature Distributions")
    
    # Create a subplot for each feature
    for feature in features:
        fig = px.histogram(
            data, 
            x=feature,
            marginal="box",
            title=f"Distribution of {feature}",
            opacity=0.7
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def show_box_plots(data, features):
    """Show box plots for selected features."""
    st.subheader("Box Plots")
    
    if st.session_state.target_column and st.session_state.target_column in data.columns:
        target_col = st.session_state.target_column
        
        # Limit to top classes if there are too many
        top_classes = 5
        value_counts = data[target_col].value_counts()
        
        if len(value_counts) > top_classes:
            st.info(f"Showing only the top {top_classes} classes due to the large number of classes.")
            selected_classes = value_counts.index[:top_classes].tolist()
            filtered_data = data[data[target_col].isin(selected_classes)]
        else:
            filtered_data = data
            selected_classes = value_counts.index.tolist()
        
        # Create box plots for each feature grouped by target
        for feature in features:
            fig = px.box(
                filtered_data,
                x=target_col,
                y=feature,
                color=target_col,
                title=f"Box Plot of {feature} by {target_col}",
                category_orders={target_col: selected_classes}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select a target column in the Data Overview tab to see box plots by class.")
        
        # Show simple box plots without grouping
        fig = px.box(
            data,
            y=features,
            title="Box Plots of Selected Features"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

def show_scatter_plot(data, features):
    """Show scatter plot for selected features."""
    st.subheader("Scatter Plot")
    
    if len(features) < 2:
        st.warning("Please select at least 2 features for scatter plot.")
        return
    
    # Select features for scatter plot
    x_feature = st.selectbox("Select X-axis feature", options=features, index=0)
    y_feature = st.selectbox("Select Y-axis feature", options=features, index=min(1, len(features)-1))
    
    # Create scatter plot
    if st.session_state.target_column and st.session_state.target_column in data.columns:
        target_col = st.session_state.target_column
        
        # Limit to top classes if there are too many
        top_classes = 10
        value_counts = data[target_col].value_counts()
        
        if len(value_counts) > top_classes:
            selected_classes = value_counts.index[:top_classes].tolist()
            filtered_data = data[data[target_col].isin(selected_classes)]
            st.info(f"Showing only the top {top_classes} classes due to the large number of classes.")
        else:
            filtered_data = data
        
        fig = px.scatter(
            filtered_data,
            x=x_feature,
            y=y_feature,
            color=target_col,
            opacity=0.7,
            title=f"Scatter Plot: {x_feature} vs {y_feature}",
            hover_data=['index'] if 'index' in filtered_data.columns else None
        )
    else:
        fig = px.scatter(
            data,
            x=x_feature,
            y=y_feature,
            opacity=0.7,
            title=f"Scatter Plot: {x_feature} vs {y_feature}",
            hover_data=['index'] if 'index' in data.columns else None
        )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def show_preprocessing_tab(data):
    """Show data preprocessing options and apply preprocessing."""
    st.header("Data Preprocessing")
    
    # Check if target column is selected
    if not st.session_state.target_column:
        st.warning("Please select a target column in the Data Overview tab before preprocessing.")
        return
    
    # Initialize preprocessor if not already in session state
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = DataPreprocessor()
    
    # Preprocessing options
    st.subheader("Preprocessing Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        handle_missing = st.checkbox("Handle Missing Values", value=True)
        remove_duplicates = st.checkbox("Remove Duplicate Rows", value=True)
        handle_outliers = st.checkbox("Handle Outliers", value=False)
    
    with col2:
        encode_categorical = st.checkbox("Encode Categorical Features", value=True)
        scale_features = st.checkbox("Scale Features", value=True)
    
    # Apply preprocessing button
    if st.button("Apply Preprocessing"):
        try:
            with st.spinner("Preprocessing data..."):
                # Copy the data to avoid modifying the original
                processed_data = data.copy()
                
                # Remove duplicates if selected
                if remove_duplicates:
                    initial_rows = len(processed_data)
                    processed_data = processed_data.drop_duplicates().reset_index(drop=True)
                    removed_duplicates = initial_rows - len(processed_data)
                    st.info(f"Removed {removed_duplicates} duplicate rows")
                
                # Clean the data (handle missing values)
                if handle_missing:
                    processed_data = st.session_state.preprocessor.clean_data(processed_data, st.session_state.target_column)
                    st.info("Handled missing values and cleaned data")
                
                # Handle outliers if selected
                if handle_outliers:
                    # Simple outlier handling (for demonstration)
                    numeric_cols = processed_data.select_dtypes(include=['float64', 'int64']).columns
                    for col in numeric_cols:
                        if col != st.session_state.target_column:
                            # Use IQR method
                            Q1 = processed_data[col].quantile(0.25)
                            Q3 = processed_data[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            
                            # Replace outliers with bounds
                            processed_data.loc[processed_data[col] < lower_bound, col] = lower_bound
                            processed_data.loc[processed_data[col] > upper_bound, col] = upper_bound
                    
                    st.info("Handled outliers using IQR method")
                
                # Encode categorical features
                if encode_categorical:
                    processed_data = st.session_state.preprocessor.encode_categorical(processed_data, st.session_state.target_column)
                    st.info("Encoded categorical features")
                
                # Scale features if selected
                if scale_features:
                    # Get features and target
                    X_scaled, y_encoded = st.session_state.preprocessor.scale_features(processed_data, st.session_state.target_column)
                    
                    # Create DataFrame from scaled features
                    feature_names = processed_data.drop(columns=[st.session_state.target_column]).columns
                    scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
                    
                    # Add target column back
                    scaled_df[st.session_state.target_column] = y_encoded
                    
                    processed_data = scaled_df
                    st.info("Scaled features using StandardScaler")
                
                # Store the preprocessed data in session state
                st.session_state.preprocessed_data = processed_data
                
                st.success("Data preprocessing completed successfully!")
                
                # Show data statistics before and after
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Before Preprocessing")
                    st.write(f"Shape: {data.shape}")
                    st.write(f"Missing values: {data.isnull().sum().sum()}")
                    st.write(f"Duplicates: {data.duplicated().sum()}")
                
                with col2:
                    st.subheader("After Preprocessing")
                    st.write(f"Shape: {processed_data.shape}")
                    st.write(f"Missing values: {processed_data.isnull().sum().sum()}")
                    st.write(f"Duplicates: {processed_data.duplicated().sum()}")
                
                # Preview preprocessed data
                st.subheader("Preprocessed Data Preview")
                st.dataframe(processed_data.head())
        
        except Exception as e:
            st.error(f"Error during preprocessing: {str(e)}")
    
    # Display existing preprocessed data if available
    if st.session_state.preprocessed_data is not None:
        st.subheader("Current Preprocessed Data")
        st.info(f"Preprocessed data shape: {st.session_state.preprocessed_data.shape}")
        st.dataframe(st.session_state.preprocessed_data.head())

def show_feature_engineering_tab():
    """Show feature engineering options and apply feature engineering."""
    st.header("Feature Engineering")
    
    # Check if preprocessed data is available
    if 'preprocessed_data' not in st.session_state or st.session_state.preprocessed_data is None:
        st.warning("Please complete data preprocessing before feature engineering.")
    
        # Add this button to continue anyway
        if st.button("Continue Anyway (Use Original Data)"):
            # Use the original data instead
            st.session_state.preprocessed_data = st.session_state.data
            st.success("Using original data for feature engineering.")
            st.experimental_rerun()
        return
        
        
    # Import feature engineer only if needed
    from src.feature_engineering import FeatureEngineer
    
    # Initialize feature engineer if not already in session state
    if 'feature_engineer' not in st.session_state:
        st.session_state.feature_engineer = FeatureEngineer()
    
    # Feature engineering options
    st.subheader("Feature Engineering Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        create_rate_features = st.checkbox("Create Rate-based Features", value=True)
        create_ratio_features = st.checkbox("Create Ratio Features", value=True)
        create_interaction_features = st.checkbox("Create Interaction Features", value=True)
    
    with col2:
        feature_selection = st.checkbox("Apply Feature Selection", value=True)
        
        if feature_selection:
            selection_method = st.selectbox(
                "Feature Selection Method",
                options=["anova", "mutual_info", "pca", "combined"]
            )
            
            n_features = st.slider(
                "Number of Features to Select",
                min_value=5,
                max_value=50,
                value=20,
                step=5
            )
    
    # Apply feature engineering button
    if st.button("Apply Feature Engineering"):
        try:
            with st.spinner("Engineering features..."):
                # Get data for feature engineering
                data = st.session_state.preprocessed_data
                
                # Separate features and target
                X = data.drop(columns=[st.session_state.target_column]).values
                y = data[st.session_state.target_column].values
                feature_names = data.drop(columns=[st.session_state.target_column]).columns.tolist()
                
                # Apply feature engineering
                X_engineered, engineered_feature_names = st.session_state.feature_engineer.engineer_features(X, feature_names)
                
                # Store engineered features information
                st.session_state.engineered_feature_count = len(engineered_feature_names) - len(feature_names)
                
                # Apply feature selection if selected
                if feature_selection:
                    X_selected, selected_feature_names = st.session_state.feature_engineer.select_features(
                        X_engineered, y, engineered_feature_names, 
                        method=selection_method, k=n_features
                    )
                    
                    # Create DataFrame with selected features
                    selected_df = pd.DataFrame(X_selected, columns=selected_feature_names)
                    
                    # Add target column back
                    selected_df[st.session_state.target_column] = y
                    
                    # Store in session state
                    st.session_state.features = selected_df
                    st.session_state.selected_feature_names = selected_feature_names
                    
                    st.success(f"Feature engineering and selection completed! Selected {len(selected_feature_names)} features.")
                else:
                    # Create DataFrame with all engineered features
                    engineered_df = pd.DataFrame(X_engineered, columns=engineered_feature_names)
                    
                    # Add target column back
                    engineered_df[st.session_state.target_column] = y
                    
                    # Store in session state
                    st.session_state.features = engineered_df
                    st.session_state.selected_feature_names = engineered_feature_names
                    
                    st.success(f"Feature engineering completed! Created {st.session_state.engineered_feature_count} new features.")
                
                # Show feature statistics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Before Feature Engineering")
                    st.write(f"Number of features: {len(feature_names)}")
                
                with col2:
                    st.subheader("After Feature Engineering")
                    if feature_selection:
                        st.write(f"Number of selected features: {len(selected_feature_names)}")
                    else:
                        st.write(f"Number of features: {len(engineered_feature_names)}")
                    st.write(f"New features created: {st.session_state.engineered_feature_count}")
                
                # Show feature importance if available
                if hasattr(st.session_state.feature_engineer, 'feature_selector') and st.session_state.feature_engineer.feature_selector is not None:
                    if hasattr(st.session_state.feature_engineer.feature_selector, 'scores_'):
                        scores = st.session_state.feature_engineer.feature_selector.scores_
                        
                        if feature_selection:
                            feature_importance = pd.DataFrame({
                                'Feature': selected_feature_names,
                                'Importance': scores[st.session_state.feature_engineer.feature_selector.get_support(indices=True)]
                            })
                        else:
                            feature_importance = pd.DataFrame({
                                'Feature': engineered_feature_names,
                                'Importance': scores
                            })
                        
                        feature_importance = feature_importance.sort_values('Importance', ascending=False)
                        
                        st.subheader("Feature Importance")
                        
                        fig = px.bar(
                            feature_importance.head(20),
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title="Top 20 Features by Importance"
                        )
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Preview final features
                st.subheader("Engineered Features Preview")
                st.dataframe(st.session_state.features.head())
        
        except Exception as e:
            st.error(f"Error during feature engineering: {str(e)}")
    
    # Display existing features if available
    if 'features' in st.session_state and st.session_state.features is not None:
        st.subheader("Current Engineered Features")
        st.info(f"Engineered data shape: {st.session_state.features.shape}")
        st.dataframe(st.session_state.features.head())

if __name__ == "__main__":
    # For testing the page in isolation
    import streamlit as st
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'preprocessed_data' not in st.session_state:
        st.session_state.preprocessed_data = None
    if 'target_column' not in st.session_state:
        st.session_state.target_column = None
    if 'features' not in st.session_state:
        st.session_state.features = None
    
    # Display the page
    show_data_page()