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
from src.feature_engineering import FeatureEngineer

def show_data_page():
    """Display the data upload and exploration page with improved UI."""
    st.markdown("<h1 style='text-align: center; color: white;'>Data Upload & Exploration</h1>", unsafe_allow_html=True)
    
    tab_style = """
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: var(--background-secondary);
        padding: 10px 20px 0 20px;
        border-radius: 15px 15px 0 0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border: none;
    }
    .stTabs [data-baseweb="tab-list"] button {
        color: #f8f9fa !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: transparent;
        border-bottom: 2px solid #4a90e2;
    }
</style>
"""
    st.markdown(tab_style, unsafe_allow_html=True)
    
    # Create tabs for the data workflow with improved styling
    tabs = st.tabs([
        "📤 Upload Data", 
        "📊 Data Overview", 
        "🔍 Feature Analysis", 
        "🧹 Preprocessing", 
        "🔧 Feature Engineering"
    ])
    
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
                st.info("Please upload data in the 'Upload Data' tab first.")

def show_data_upload_tab():
    """Show data upload functionality with improved UI."""
    st.markdown("<h2>Upload Network Traffic Data</h2>", unsafe_allow_html=True)
    
    # Create centered file uploader
    upload_container_style = """
    <style>
        [data-testid="stFileUploader"] {
            background-color: var(--background-secondary);
            padding: 2rem;
            border-radius: 10px;
            border: 1px dashed #444;
            margin-bottom: 2rem;
        }
        [data-testid="stFileUploader"] button {
            background-color: var(--background-secondary);
            color: white;
        }
    </style>
    """
    st.markdown(upload_container_style, unsafe_allow_html=True)
    
    # Create a better layout for upload options
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Improved file uploader with instructions
        st.markdown("""
        <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 10px; border-left: 4px solid #4a90e2;'>
            <h4 style='margin-top: 0;'>CSV Upload Instructions</h4>
            <p>Select a CSV file containing network traffic data. The file should contain network flow features and ideally have a target column with attack labels.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # File upload with expanded area
        uploaded_file = st.file_uploader(
            "Choose a CSV file with network traffic data", 
            type="csv",
            help="Upload a CSV file containing network traffic data with features and labels"
        )
        
        # Process the uploaded file
        if uploaded_file is not None:
            try:
                with st.spinner("Loading and analyzing data..."):
                    # Add progress bar for better user feedback
                    progress_bar = st.progress(0)
                    
                    # Update progress
                    progress_bar.progress(25)
                    time.sleep(0.5)  # Simulate processing time
                    
                    # Load the data
                    data = pd.read_csv(uploaded_file)
                    
                    # Convert numeric columns
                    for col in data.columns:
                        if str(data[col].dtype).startswith(('Int', 'Float')):
                            data[col] = data[col].astype(float)
                    
                    # Update progress
                    progress_bar.progress(50)
                    time.sleep(0.5)  # Simulate processing time
                    
                    # Store in session state
                    st.session_state.data = data
                    
                    # Reset other session state variables that depend on the data
                    st.session_state.preprocessed_data = None
                    st.session_state.features = None
                    st.session_state.trained_models = {}
                    st.session_state.best_model = None
                    st.session_state.evaluation_results = None
                    st.session_state.predictions = None
                    
                    # Update progress
                    progress_bar.progress(75)
                    time.sleep(0.5)  # Simulate processing time
                    
                    # Try to identify target column
                    potential_target = identify_target_column(data)
                    st.session_state.target_column = potential_target
                    
                    # Final progress update
                    progress_bar.progress(100)
                    
                    # Success message with enhanced styling
                    st.success(f"✅ Successfully loaded data with {data.shape[0]:,} rows and {data.shape[1]:,} columns!")
                    
                    # Show metadata about the dataset
                    st.markdown(f"""
                    <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
                        <h4 style='margin-top: 0;'>Dataset Information</h4>
                        <ul>
                            <li><strong>Rows:</strong> {data.shape[0]:,}</li>
                            <li><strong>Columns:</strong> {data.shape[1]:,}</li>
                            <li><strong>Memory Usage:</strong> {data.memory_usage(deep=True).sum() / (1024**2):.2f} MB</li>
                            <li><strong>Numeric Features:</strong> {len(data.select_dtypes(include=['int64', 'float64']).columns)}</li>
                            <li><strong>Categorical Features:</strong> {len(data.select_dtypes(include=['object']).columns)}</li>
                            <li><strong>Missing Values:</strong> {data.isnull().sum().sum():,}</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show quick preview
                    st.subheader("Data Preview")
                    st.dataframe(data.head(10), use_container_width=True)
                    
                    # Prompt to go to next tab
                    st.info("👉 Now go to the 'Data Overview' tab to explore your dataset")
            
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    with col2:
        # Sample dataset option with improved card styling
        st.markdown("""
        <div style='background-color: var(--background-secondary); padding: 1.2rem; border-radius: 10px; height: 100%;'>
            <h3 style='margin-top: 0; text-align: center;'>🧪 Sample Dataset</h3>
            <p>Don't have your own data? Use our sample network traffic dataset containing various attack types.</p>
            <ul>
                <li>Port scan attacks</li>
                <li>DDoS attacks</li>
                <li>Normal traffic</li>
                <li>Web attacks</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Center the button
        col2_1, col2_2, col2_3 = st.columns([1, 2, 1])
        with col2_2:
            sample_button_style = """
                <style>
                div[data-testid="stButton"] button {
                background-color: var(--background-secondary);
                color: white;
                border-radius: 6px;
                border: none;
                padding: 0.6rem 1.2rem;
                width: 100%;
                font-weight: 500;
                }
            </style>
            """
            st.markdown(sample_button_style, unsafe_allow_html=True)
            
            use_sample = st.button("Load Sample Dataset", use_container_width=True)
        
        if use_sample:
            try:
                # Check if sample data exists
                sample_path = "data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
                if os.path.exists(sample_path):
                    with st.spinner("Loading sample dataset..."):
                        # Add progress bar
                        progress_bar = st.progress(0)
                        
                        for i in range(0, 101, 20):
                            progress_bar.progress(i)
                            time.sleep(0.2)  # Simulate loading time
                        
                        data = pd.read_csv(sample_path)
                        st.session_state.data = data
                        st.session_state.target_column = ' Label' if ' Label' in data.columns else None
                        
                        progress_bar.progress(100)
                        st.success(f"✅ Loaded sample dataset with {data.shape[0]:,} rows and {data.shape[1]:,} columns!")
                        
                        # Prompt to go to next tab
                        st.info("👉 Now go to the 'Data Overview' tab to explore the dataset")
                else:
                    st.error("❌ Sample dataset not found. Please upload your own data.")
            except Exception as e:
                st.error(f"Error loading sample dataset: {str(e)}")
                
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
    """Show data overview with improved UI."""
    st.markdown("<h2>Dataset Overview</h2>", unsafe_allow_html=True)
    
    # Data summary cards
    with st.container():
        st.markdown("<h3>Key Statistics</h3>", unsafe_allow_html=True)
        
        # Display metrics in a grid
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", f"{data.shape[0]:,}")
        
        with col2:
            st.metric("Total Columns", f"{data.shape[1]:,}")
        
        with col3:
            st.metric("Missing Values", f"{data.isnull().sum().sum():,}")
        
        with col4:
            st.metric("Duplicates", f"{data.duplicated().sum():,}")
    
    # Column types visualization
    with st.container():
        st.markdown("<h3>Data Types Distribution</h3>", unsafe_allow_html=True)
        
        # Get column type counts
        dtype_counts = data.dtypes.value_counts().reset_index()
        dtype_counts.columns = ['Data Type', 'Count']
        
        # Prepare for plotting
        dtype_counts = prepare_dataframe_for_plotting(dtype_counts)
        
        # Create pie chart for data types
        fig = px.pie(
            dtype_counts, 
            values='Count',
            names='Data Type',
            color_discrete_sequence=px.colors.qualitative.Safe,
            title="Distribution of Column Data Types"
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Target column selection with improved UI
    with st.container():
        st.markdown("<h3>Target Column Selection</h3>", unsafe_allow_html=True)
        
        # Info about target column
        st.markdown("""
        <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
            <p>The target column contains the labels for network traffic classification (normal vs. attack types).</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get potential target columns
        target_columns = []
        for col in data.columns:
            if data[col].nunique() <= 20:  # Only consider columns with reasonable number of classes
                target_columns.append(col)
        
        # Default to the automatically identified target column
        default_idx = 0
        if st.session_state.target_column in target_columns:
            default_idx = target_columns.index(st.session_state.target_column)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Target selection with better description
            selected_target = st.selectbox(
                "Select the column containing traffic classification labels",
                options=target_columns,
                index=default_idx,
                help="This column should identify normal traffic vs. different attack types"
            )
        
        with col2:
            # Set target button
            if st.button("Set as Target", use_container_width=True):
                st.session_state.target_column = selected_target
                st.success(f"✅ Set '{selected_target}' as the target column!")
    
    # Show class distribution if target column is selected
    if st.session_state.target_column:
        display_class_distribution(data, st.session_state.target_column)

def display_class_distribution(data, target_col):
    """Display class distribution with improved visualization."""
    st.markdown("<h3>Class Distribution</h3>", unsafe_allow_html=True)
    
    # Get class counts
    class_counts = data[target_col].value_counts().reset_index()
    class_counts.columns = ['Class', 'Count']
    class_counts['Percentage'] = 100 * class_counts['Count'] / class_counts['Count'].sum()
    
    # Prepare for plotting
    class_counts = prepare_dataframe_for_plotting(class_counts)
    
    # Create visualization columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create bar chart
        fig = px.bar(
            class_counts,
            x='Class',
            y='Count',
            color='Class',
            text=class_counts['Percentage'].apply(lambda x: f'{x:.1f}%'),
            title=f"Distribution of {target_col}",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        
        fig.update_layout(
            xaxis={'categoryorder':'total descending'},
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Display class balance information
        st.markdown("<h4>Class Balance Analysis</h4>", unsafe_allow_html=True)
        
        # Determine if dataset is imbalanced
        max_class = class_counts['Count'].max()
        min_class = class_counts['Count'].min()
        imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
        
        # Majority and minority classes
        majority_class = class_counts.loc[class_counts['Count'].idxmax(), 'Class']
        minority_class = class_counts.loc[class_counts['Count'].idxmin(), 'Class']
        
        # Color based on imbalance
        balance_color = "#28a745" if imbalance_ratio < 10 else "#ffc107" if imbalance_ratio < 100 else "#dc3545"
        
        st.markdown(f"""
        <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
            <p><strong>Number of Classes:</strong> {len(class_counts)}</p>
            <p><strong>Imbalance Ratio:</strong> <span style='color: {balance_color};'>{imbalance_ratio:.2f}</span></p>
            <p><strong>Majority Class:</strong> {majority_class} ({class_counts.loc[class_counts['Count'].idxmax(), 'Percentage']:.1f}%)</p>
            <p><strong>Minority Class:</strong> {minority_class} ({class_counts.loc[class_counts['Count'].idxmin(), 'Percentage']:.1f}%)</p>
        </div>
        
        <div style='margin-top: 1rem;'>
            <h4>Balance Assessment</h4>
            <p>
                {
                "✅ The dataset is well-balanced." if imbalance_ratio < 10 else
                "⚠️ The dataset is moderately imbalanced. Consider balancing techniques during model training." if imbalance_ratio < 100 else
                "❌ The dataset is highly imbalanced. Balancing techniques are strongly recommended."
                }
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Show detailed table
    with st.expander("Show detailed class distribution data"):
        st.dataframe(class_counts, use_container_width=True)
        
def show_feature_analysis_tab(data):
    """Show feature analysis with improved UI."""
    st.markdown("<h2>Feature Analysis</h2>", unsafe_allow_html=True)
    
    # Create better layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Feature selection section with improved styling
        st.markdown("""
        <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
            <h3 style='margin-top: 0;'>Select Features</h3>
            <p>Choose features to analyze their distributions and relationships.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get numerical columns for analysis
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if not numeric_cols:
            st.warning("No numerical features found for analysis.")
            return
        
        # Select features to analyze with better UI
        selected_features = st.multiselect(
            "Select features to analyze",
            options=numeric_cols,
            default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols,
            help="Choose features to explore their distributions and relationships"
        )
        
        if not selected_features:
            st.info("Please select at least one feature to analyze.")
            return
        
        # Improved visualization selector
        st.markdown("<h4>Visualization Type</h4>", unsafe_allow_html=True)
        
        viz_type = st.radio(
            "Select visualization type",
            ["Correlation Heatmap", "Feature Distributions", "Box Plots", "Scatter Plot"],
            help="Choose how to visualize the selected features"
        )
    
    with col2:
        # Display selected visualization with improved styling
        if not selected_features:
            st.info("Select features from the sidebar to visualize them here.")
            return
            
        if viz_type == "Correlation Heatmap":
            st.markdown("<h3>Correlation Analysis</h3>", unsafe_allow_html=True)
            show_correlation_heatmap(data, selected_features)
        
        elif viz_type == "Feature Distributions":
            st.markdown("<h3>Feature Distributions</h3>", unsafe_allow_html=True)
            show_feature_distributions(data, selected_features)
        
        elif viz_type == "Box Plots":
            st.markdown("<h3>Box Plot Analysis</h3>", unsafe_allow_html=True)
            show_box_plots(data, selected_features)
        
        elif viz_type == "Scatter Plot":
            st.markdown("<h3>Feature Relationships</h3>", unsafe_allow_html=True)
            show_scatter_plot(data, selected_features)

def show_correlation_heatmap(data, features):
    """Show correlation heatmap with improved styling."""
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
        aspect="auto",
        title="Feature Correlation Matrix"
    )
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Identify highly correlated features
    threshold = 0.8
    high_corr = []
    
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr.append((features[i], features[j], corr_matrix.iloc[i, j]))
    
    if high_corr:
        st.markdown("<h4>Highly Correlated Feature Pairs</h4>", unsafe_allow_html=True)
        
        # Create a table for correlated features
        corr_data = []
        for feat1, feat2, corr in high_corr:
            corr_data.append({
                "Feature 1": feat1,
                "Feature 2": feat2,
                "Correlation": corr,
                "Relationship": "Strong Positive" if corr > 0.8 else "Strong Negative"
            })
        
        corr_df = pd.DataFrame(corr_data)
        st.dataframe(corr_df, use_container_width=True)
        
        # Add insight about feature redundancy
        st.markdown("""
        <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
            <h4 style='margin-top: 0;'>⚠️ Possible Feature Redundancy</h4>
            <p>The highlighted feature pairs have high correlation (>0.8), suggesting potential redundancy.
            Consider removing one feature from each pair during feature selection to reduce dimensionality and improve model performance.</p>
        </div>
        """, unsafe_allow_html=True)

def show_feature_distributions(data, features):
    """Show distribution plots with improved styling."""
    # Create a subplot for each feature
    for feature in features:
        # Create an expander for each feature
        with st.expander(f"Distribution of {feature}", expanded=True if features.index(feature) == 0 else False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Create distribution plot
                fig = px.histogram(
                    data, 
                    x=feature,
                    marginal="box",
                    opacity=0.7,
                    color_discrete_sequence=['#4a90e2'],
                    title=f"Distribution of {feature}"
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Show statistics
                stats = data[feature].describe()
                
                # Detect outliers
                Q1 = stats['25%']
                Q3 = stats['75%']
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)][feature]
                outlier_pct = 100 * len(outliers) / len(data)
                
                # Determine distribution skewness
                skewness = data[feature].skew()
                skew_type = "Symmetric" if abs(skewness) < 0.5 else "Moderately Skewed" if abs(skewness) < 1 else "Highly Skewed"
                skew_direction = "Positive (right tail)" if skewness > 0 else "Negative (left tail)" if skewness < 0 else ""
                
                st.markdown(f"""
                <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 10px;'>
                    <h4 style='margin-top: 0;'>Statistics</h4>
                    <table style='width: 100%;'>
                        <tr><td><strong>Mean:</strong></td><td>{stats['mean']:.4f}</td></tr>
                        <tr><td><strong>Median:</strong></td><td>{stats['50%']:.4f}</td></tr>
                        <tr><td><strong>Std Dev:</strong></td><td>{stats['std']:.4f}</td></tr>
                        <tr><td><strong>Min:</strong></td><td>{stats['min']:.4f}</td></tr>
                        <tr><td><strong>Max:</strong></td><td>{stats['max']:.4f}</td></tr>
                    </table>
                    
                    <h4 style='margin-top: 1rem;'>Analysis</h4>
                    <p><strong>Skewness:</strong> {skewness:.2f} ({skew_type} {skew_direction})</p>
                    <p><strong>Outliers:</strong> {len(outliers):,} ({outlier_pct:.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)
                
def show_box_plots(data, features):
    """Show box plots with improved styling."""
    # Check if target column is selected
    if st.session_state.target_column and st.session_state.target_column in data.columns:
        target_col = st.session_state.target_column
        
        # Create selectbox for feature selection
        if len(features) > 1:
            selected_feature = st.selectbox(
                "Select feature to analyze",
                options=features
            )
        else:
            selected_feature = features[0]
        
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
        
        # Create box plot for the selected feature
        fig = px.box(
            filtered_data,
            x=target_col,
            y=selected_feature,
            color=target_col,
            title=f"Distribution of {selected_feature} by {target_col}",
            color_discrete_sequence=px.colors.qualitative.Safe,
            category_orders={target_col: selected_classes},
            notched=True
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add insights about the distribution
        st.markdown("<h4>Distribution Insights</h4>", unsafe_allow_html=True)
        
        # Calculate statistics for each class
        class_stats = []
        for cls in selected_classes:
            class_data = filtered_data[filtered_data[target_col] == cls][selected_feature]
            
            stats = {
                "Class": cls,
                "Mean": class_data.mean(),
                "Median": class_data.median(),
                "Std Dev": class_data.std(),
                "Min": class_data.min(),
                "Max": class_data.max()
            }
            
            class_stats.append(stats)
        
        # Create a dataframe of class statistics
        stats_df = pd.DataFrame(class_stats)
        st.dataframe(stats_df, use_container_width=True)
        
        # Check if the feature is good for class separation
        means = stats_df['Mean'].values
        stds = stats_df['Std Dev'].values
        
        # Calculate coefficient of variation of means
        cv_means = np.std(means) / np.mean(means) if np.mean(means) != 0 else 0
        
        # Calculate average overlap
        avg_std = np.mean(stds)
        mean_range = np.max(means) - np.min(means)
        
        separation_score = mean_range / avg_std if avg_std > 0 else float('inf')
        
        separation_quality = (
            "Excellent" if separation_score > 2 else
            "Good" if separation_score > 1 else
            "Fair" if separation_score > 0.5 else
            "Poor"
        )
        
        separation_color = (
            "#28a745" if separation_score > 2 else
            "#4a90e2" if separation_score > 1 else
            "#ffc107" if separation_score > 0.5 else
            "#dc3545"
        )
        
        st.markdown(f"""
        <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
            <h4 style='margin-top: 0;'>Feature Separation Analysis</h4>
            <p><strong>Separation Score:</strong> <span style='color: {separation_color};'>{separation_score:.2f}</span></p>
            <p><strong>Separation Quality:</strong> <span style='color: {separation_color};'>{separation_quality}</span></p>
            <p><strong>Insight:</strong> This feature is {
                "likely to be very useful for classification" if separation_score > 2 else
                "potentially useful for classification" if separation_score > 1 else
                "somewhat useful but may need feature engineering" if separation_score > 0.5 else
                "not likely to be useful on its own for classification"
            }</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Please select a target column in the Data Overview tab to see box plots by class.")
        
        # Show simple box plots without grouping
        fig = px.box(
            data,
            y=features,
            title="Box Plots of Selected Features",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

def show_scatter_plot(data, features):
    """Show scatter plot with improved styling."""
    if len(features) < 2:
        st.warning("Please select at least 2 features for scatter plot.")
        return
    
    # Select features for scatter plot
    col1, col2 = st.columns(2)
    
    with col1:
        x_feature = st.selectbox("Select X-axis feature", options=features, index=0)
    
    with col2:
        y_feature = st.selectbox("Select Y-axis feature", options=features, index=min(1, len(features)-1))
    
    # Create scatter plot
    if st.session_state.target_column and st.session_state.target_column in data.columns:
        # Use target column for coloring
        target_col = st.session_state.target_column
        
        # Limit to top classes if there are too many
        top_classes = 10
        value_counts = data[target_col].value_counts()
        
        if len(value_counts) > top_classes:
            st.info(f"Showing only the top {top_classes} classes due to the large number of classes.")
            selected_classes = value_counts.index[:top_classes].tolist()
            filtered_data = data[data[target_col].isin(selected_classes)]
        else:
            filtered_data = data
            selected_classes = value_counts.index.tolist()
        
        # Create scatter plot with target coloring
        fig = px.scatter(
            filtered_data,
            x=x_feature,
            y=y_feature,
            color=target_col,
            opacity=0.7,
            title=f"Scatter Plot: {x_feature} vs {y_feature} by Class",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add insight about cluster patterns
        st.markdown("""
        <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
            <h4 style='margin-top: 0;'>Interpretation</h4>
            <p>Look for these patterns in the scatter plot:</p>
            <ul>
                <li><strong>Distinct clusters</strong>: Well-separated clusters by color indicate the features are good for distinguishing between classes.</li>
                <li><strong>Overlapping clusters</strong>: If colors are mixed together, these features alone may not be sufficient for classification.</li>
                <li><strong>Outliers</strong>: Points far away from their cluster might be anomalies or potentially interesting attack patterns.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Create simple scatter plot without coloring
        fig = px.scatter(
            data,
            x=x_feature,
            y=y_feature,
            opacity=0.7,
            title=f"Scatter Plot: {x_feature} vs {y_feature}"
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add correlation info
        correlation = data[x_feature].corr(data[y_feature])
        
        corr_strength = (
            "Strong positive" if correlation > 0.7 else
            "Moderate positive" if correlation > 0.3 else
            "Weak positive" if correlation > 0 else
            "Weak negative" if correlation > -0.3 else
            "Moderate negative" if correlation > -0.7 else
            "Strong negative"
        )
        
        corr_color = (
            "#28a745" if correlation > 0.7 or correlation < -0.7 else
            "#4a90e2" if correlation > 0.3 or correlation < -0.3 else
            "#6c757d"
        )
        
        st.markdown(f"""
        <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
            <h4 style='margin-top: 0;'>Correlation Analysis</h4>
            <p><strong>Correlation Coefficient:</strong> <span style='color: {corr_color};'>{correlation:.2f}</span></p>
            <p><strong>Relationship:</strong> <span style='color: {corr_color};'>{corr_strength}</span> correlation between {x_feature} and {y_feature}.</p>
        </div>
        """, unsafe_allow_html=True)

def show_preprocessing_tab(data):
    """Show data preprocessing options and apply preprocessing with improved UI."""
    st.markdown("<h2>Data Preprocessing</h2>", unsafe_allow_html=True)
    
    # Check if target column is selected
    if not st.session_state.target_column:
        st.warning("⚠️ Please select a target column in the Data Overview tab before preprocessing.")
        return
    
    # Initialize preprocessor if not already in session state
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = DataPreprocessor()
    
    # Create a better layout for preprocessing
    st.markdown("""
    <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 10px; margin-bottom: 1.5rem;'>
        <h3 style='margin-top: 0;'>Preprocessing Pipeline</h3>
        <p>Configure and apply data preprocessing steps to prepare your data for machine learning models.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for preprocessing options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4>Data Cleaning</h4>", unsafe_allow_html=True)
        
        handle_missing = st.checkbox(
            "Handle Missing Values", 
            value=True,
            help="Replace missing values with appropriate strategies (mean, median, or mode)"
        )
        
        remove_duplicates = st.checkbox(
            "Remove Duplicate Rows", 
            value=True,
            help="Remove identical rows to prevent biased training"
        )
        
        handle_outliers = st.checkbox(
            "Handle Outliers", 
            value=False,
            help="Use IQR method to detect and handle outliers (trim or cap extreme values)"
        )
        
        # Add outlier handling options if selected
        if handle_outliers:
            outlier_method = st.radio(
                "Outlier Handling Method",
                options=["Cap at Percentile", "Remove Outliers"],
                help="Choose how to handle detected outliers"
            )
            
            if outlier_method == "Cap at Percentile":
                outlier_percentile = st.slider(
                    "Percentile Threshold",
                    min_value=95,
                    max_value=99,
                    value=97,
                    step=1,
                    help="Values beyond this percentile will be capped"
                )
    
    with col2:
        st.markdown("<h4>Feature Transformation</h4>", unsafe_allow_html=True)
        
        encode_categorical = st.checkbox(
            "Encode Categorical Features", 
            value=True,
            help="Convert categorical variables to numerical representation"
        )
        
        # Add encoding method if selected
        if encode_categorical:
            encoding_method = st.radio(
                "Encoding Method",
                options=["One-Hot Encoding", "Label Encoding"],
                help="One-hot creates binary columns, Label assigns numeric values"
            )
        
        scale_features = st.checkbox(
            "Scale Features", 
            value=True,
            help="Normalize or standardize feature values to improve model performance"
        )
        
        # Add scaling method if selected
        if scale_features:
            scaling_method = st.radio(
                "Scaling Method",
                options=["StandardScaler", "MinMaxScaler", "RobustScaler"],
                help="Standard scales to mean=0 and std=1, MinMax scales to 0-1 range"
            )
        # Visualization of preprocessing steps
    st.markdown("<h4>Preprocessing Pipeline Visualization</h4>", unsafe_allow_html=True)
    
    # Render pipeline steps
    active_steps = []
    if remove_duplicates:
        active_steps.append({"name": "Remove Duplicates", "desc": "Remove identical rows"})
    if handle_missing:
        active_steps.append({"name": "Handle Missing Values", "desc": "Replace NaN values"})
    if handle_outliers:
        active_steps.append({"name": "Handle Outliers", "desc": f"{outlier_method} for extreme values"})
    if encode_categorical:
        active_steps.append({"name": "Encode Categorical", "desc": f"Using {encoding_method}"})
    if scale_features:
        active_steps.append({"name": "Scale Features", "desc": f"Using {scaling_method}"})
    
    # Render pipeline flow
    st.markdown("""
    <style>
    .pipeline-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 20px 0;
        padding: 0 10px;
    }
    .pipeline-step {
        background-color: var(--background-secondary);
        color: white;
        border-radius: 8px;
        padding: 10px 15px;
        text-align: center;
        min-width: 120px;
        position: relative;
        font-weight: 500;
    }
    .pipeline-step::after {
        content: "→";
        position: absolute;
        right: -20px;
        top: 50%;
        transform: translateY(-50%);
        color: #6c757d;
        font-size: 1.5rem;
    }
    .pipeline-step:last-child::after {
        content: "";
    }
    .pipeline-step-desc {
        font-size: 0.8rem;
        margin-top: 5px;
        font-weight: normal;
    }
    </style>
    <div class="pipeline-container">
        <div class="pipeline-step" style="background-color: var(--background-secondary);">
            Raw Data
            <div class="pipeline-step-desc">Unprocessed</div>
        </div>
        """ + "".join([f"""
        <div class="pipeline-step">
            {step["name"]}
            <div class="pipeline-step-desc">{step["desc"]}</div>
        </div>
        """ for step in active_steps]) + f"""
        <div class="pipeline-step" style="background-color: var(--background-secondary);">
            Processed Data
            <div class="pipeline-step-desc">Ready for modeling</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Apply preprocessing button
    if st.button("Apply Preprocessing", use_container_width=True):
        try:
            with st.spinner("Preprocessing data..."):
                # Create a progress bar
                progress_bar = st.progress(0)
                
                # Copy the data to avoid modifying the original
                processed_data = data.copy()
                
                # Track preprocessing steps for reporting
                preprocessing_steps = []
                
                # Update progress
                progress_bar.progress(10)
                time.sleep(0.2)  # Simulate processing time
                
                # Remove duplicates if selected
                if remove_duplicates:
                    initial_rows = len(processed_data)
                    processed_data = processed_data.drop_duplicates().reset_index(drop=True)
                    removed_duplicates = initial_rows - len(processed_data)
                    preprocessing_steps.append(f"Removed {removed_duplicates:,} duplicate rows")
                
                # Update progress
                progress_bar.progress(30)
                time.sleep(0.2)  # Simulate processing time
                
                # Clean the data (handle missing values)
                if handle_missing:
                    initial_na = processed_data.isna().sum().sum()
                    processed_data = st.session_state.preprocessor.clean_data(processed_data, st.session_state.target_column)
                    remaining_na = processed_data.isna().sum().sum()
                    preprocessing_steps.append(f"Filled {initial_na - remaining_na:,} missing values")
                
                # Update progress
                progress_bar.progress(50)
                time.sleep(0.2)  # Simulate processing time
                
                # Handle outliers if selected
                if handle_outliers:
                    outlier_count = 0
                    
                    # Define numeric columns
                    numeric_cols = processed_data.select_dtypes(include=['float64', 'int64']).columns
                    numeric_cols = [col for col in numeric_cols if col != st.session_state.target_column]
                    
                    for col in numeric_cols:
                        # Calculate IQR
                        Q1 = processed_data[col].quantile(0.25)
                        Q3 = processed_data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        # Count outliers
                        outliers = processed_data[(processed_data[col] < lower_bound) | (processed_data[col] > upper_bound)][col]
                        outlier_count += len(outliers)
                        
                        if outlier_method == "Cap at Percentile":
                            # Cap at percentiles
                            lower_cap = processed_data[col].quantile(0.01)
                            upper_cap = processed_data[col].quantile(outlier_percentile / 100)
                            
                            processed_data[col] = processed_data[col].clip(lower=lower_cap, upper=upper_cap)
                        else:  # Remove outliers
                            # Create a mask for non-outliers
                            mask = (processed_data[col] >= lower_bound) & (processed_data[col] <= upper_bound)
                            processed_data = processed_data[mask].reset_index(drop=True)
                    
                    preprocessing_steps.append(f"Handled {outlier_count:,} outliers using {outlier_method}")
                    
                    # Update progress
                progress_bar.progress(70)
                time.sleep(0.2)  # Simulate processing time
                
                # Encode categorical features
                if encode_categorical:
                    if encoding_method == "One-Hot Encoding":
                        # Define categorical columns
                        cat_cols = processed_data.select_dtypes(include=['object']).columns
                        cat_cols = [col for col in cat_cols if col != st.session_state.target_column]
                        
                        # Apply one-hot encoding
                        for col in cat_cols:
                            one_hot = pd.get_dummies(processed_data[col], prefix=col, drop_first=False)
                            processed_data = pd.concat([processed_data, one_hot], axis=1)
                            processed_data = processed_data.drop(col, axis=1)
                        
                        preprocessing_steps.append(f"Applied one-hot encoding to {len(cat_cols)} categorical features")
                    else:  # Label Encoding
                        processed_data = st.session_state.preprocessor.encode_categorical(processed_data, st.session_state.target_column)
                        preprocessing_steps.append("Applied label encoding to categorical features")
                
                # Update progress
                progress_bar.progress(90)
                time.sleep(0.2)  # Simulate processing time
                
                # Scale features if selected
                if scale_features:
                    # Get features and target
                    X = processed_data.drop(columns=[st.session_state.target_column]).values
                    y = processed_data[st.session_state.target_column].values
                    
                    # Scale features
                    if scaling_method == "StandardScaler":
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                    elif scaling_method == "MinMaxScaler":
                        from sklearn.preprocessing import MinMaxScaler
                        scaler = MinMaxScaler()
                    else:  # RobustScaler
                        from sklearn.preprocessing import RobustScaler
                        scaler = RobustScaler()
                    
                    # Scale features
                    X_scaled = scaler.fit_transform(X)
                    
                    # Create DataFrame from scaled features
                    feature_names = processed_data.drop(columns=[st.session_state.target_column]).columns
                    scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
                    
                    # Add target column back
                    scaled_df[st.session_state.target_column] = y
                    
                    processed_data = scaled_df
                    preprocessing_steps.append(f"Scaled features using {scaling_method}")
                
                # Store the preprocessed data in session state
                st.session_state.preprocessed_data = processed_data
                
                # Final progress update
                progress_bar.progress(100)
                
                # Success message with enhanced styling
                st.success("✅ Data preprocessing completed successfully!")
                
                # Show preprocessing report
                st.markdown("<h4>Preprocessing Report</h4>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<h5>Before Preprocessing</h5>", unsafe_allow_html=True)
                    st.write(f"Shape: {data.shape[0]:,} rows × {data.shape[1]:,} columns")
                    st.write(f"Missing values: {data.isnull().sum().sum():,}")
                    st.write(f"Duplicates: {data.duplicated().sum():,}")
                    st.write(f"Data types: {len(data.select_dtypes(include=['object']).columns)} categorical, {len(data.select_dtypes(include=['int64', 'float64']).columns)} numeric")
                
                with col2:
                    st.markdown("<h5>After Preprocessing</h5>", unsafe_allow_html=True)
                    st.write(f"Shape: {processed_data.shape[0]:,} rows × {processed_data.shape[1]:,} columns")
                    st.write(f"Missing values: {processed_data.isnull().sum().sum():,}")
                    st.write(f"Duplicates: {processed_data.duplicated().sum():,}")
                    st.write(f"Data types: {len(processed_data.select_dtypes(include=['object']).columns)} categorical, {len(processed_data.select_dtypes(include=['int64', 'float64']).columns)} numeric")
                
                # Show applied steps
                st.markdown("<h5>Applied Preprocessing Steps</h5>", unsafe_allow_html=True)
                for i, step in enumerate(preprocessing_steps, 1):
                    st.markdown(f"{i}. {step}")
                
                # Prompt to go to next step
                st.markdown("""
                <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
                    <h4 style='margin-top: 0;'>Next Step: Feature Engineering</h4>
                    <p>Your data has been successfully preprocessed. Now move to the Feature Engineering tab to create additional features for improved model performance.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Preview preprocessed data
                st.subheader("Preprocessed Data Preview")
                st.dataframe(processed_data.head(), use_container_width=True)
        
        except Exception as e:
            st.error(f"Error during preprocessing: {str(e)}")
    
    # Display existing preprocessed data if available
    if 'preprocessed_data' in st.session_state and st.session_state.preprocessed_data is not None:
        with st.expander("View Current Preprocessed Data", expanded=False):
            st.info(f"Current preprocessed data shape: {st.session_state.preprocessed_data.shape[0]:,} rows × {st.session_state.preprocessed_data.shape[1]:,} columns")
            st.dataframe(st.session_state.preprocessed_data.head(), use_container_width=True)

def show_feature_engineering_tab():
    """Show feature engineering options with improved UI."""
    st.markdown("<h2>Feature Engineering</h2>", unsafe_allow_html=True)
    
    # Check if we have preprocessed data
    if 'preprocessed_data' in st.session_state and st.session_state.preprocessed_data is not None:
        # Use preprocessed data (ideal case)
        data_to_use = st.session_state.preprocessed_data
        st.info("✅ Using preprocessed data for feature engineering.")
    elif 'data' in st.session_state and st.session_state.data is not None:
        # Fall back to original data
        data_to_use = st.session_state.data
        st.warning("⚠️ Using original data since preprocessed data is not available. This is not optimal but will allow you to continue.")
        
        # Save it as preprocessed data for future reference
        st.session_state.preprocessed_data = st.session_state.data
    else:
        # No data at all
        st.error("❌ No data available. Please upload data first and complete preprocessing.")
        return
    
    # Create a better layout
    st.markdown("""
    <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 10px; margin-bottom: 1.5rem;'>
        <h3 style='margin-top: 0;'>Feature Engineering Pipeline</h3>
        <p>Create new features to improve model performance by extracting more information from the existing data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize feature engineer if not already in session state
    if 'feature_engineer' not in st.session_state:
        st.session_state.feature_engineer = FeatureEngineer()
    
    # Feature engineering options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4>Feature Creation</h4>", unsafe_allow_html=True)
        
        create_rate_features = st.checkbox(
            "Create Rate-based Features", 
            value=True,
            help="Create rate features (e.g., bytes per packet, packets per second)"
        )
        
        create_ratio_features = st.checkbox(
            "Create Ratio Features", 
            value=True,
            help="Create ratio between related numeric features"
        )
        
        create_interaction_features = st.checkbox(
            "Create Interaction Features", 
            value=True,
            help="Create features from interactions between existing features"
        )
        
        # Add additional options
        create_polynomial_features = st.checkbox(
            "Create Polynomial Features",
            value=False,
            help="Create polynomial combinations of features (e.g., squares, cubes)"
        )
        
        if create_polynomial_features:
            poly_degree = st.slider(
                "Polynomial Degree",
                min_value=2,
                max_value=3,
                value=2,
                help="Higher degrees create more complex features but may lead to overfitting"
            )
    
    with col2:
        st.markdown("<h4>Feature Selection</h4>", unsafe_allow_html=True)
        
        feature_selection = st.checkbox(
            "Apply Feature Selection", 
            value=True,
            help="Select the most important features to reduce dimensionality"
        )
        
        if feature_selection:
            selection_method = st.selectbox(
                "Feature Selection Method",
                options=["anova", "mutual_info", "pca", "combined"],
                help="Method to rank and select features"
            )
            
            n_features = st.slider(
                "Number of Features to Select",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
                help="Number of top features to keep"
            )
            
            # Add visualization for feature selection explanation
            st.markdown("""
            <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 8px; margin-top: 1rem;'>
                <h5 style='margin-top: 0;'>Selection Methods</h5>
                <ul style='margin-bottom: 0;'>
                    <li><strong>ANOVA:</strong> Selects features based on their relationship with the target (F-test)</li>
                    <li><strong>Mutual Info:</strong> Measures dependency between features and target without linear assumption</li>
                    <li><strong>PCA:</strong> Creates new orthogonal features that capture the most variance</li>
                    <li><strong>Combined:</strong> Uses an ensemble of methods for more robust selection</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    # Apply feature engineering button
    if st.button("Apply Feature Engineering", use_container_width=True):
        try:
            with st.spinner("Engineering features..."):
                # Create a progress bar
                progress_bar = st.progress(0)
                
                # Use the preprocessed data
                data = data_to_use
                
                # Check for and handle NaN values and infinities
                has_issues = data.isna().any().any() or np.isinf(data.select_dtypes(include=['float64', 'int64'])).any().any()
                
                if has_issues:
                    st.warning("⚠️ Data contains missing values or infinity values. Fixing automatically...")
                    
                    # Make a copy to avoid modifying the original
                    data = data.copy()
                    
                    # Handle numeric columns
                    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
                    for col in numeric_cols:
                        # Get median of non-NA, non-infinite values
                        valid_values = data[col][~data[col].isna() & ~np.isinf(data[col])]
                        median_value = valid_values.median() if len(valid_values) > 0 else 0
                        
                        # Replace NaN and infinity with median
                        data[col] = data[col].replace([np.inf, -np.inf], np.nan)
                        data[col] = data[col].fillna(median_value)
                    
                    # Handle categorical columns
                    cat_cols = data.select_dtypes(include=['object']).columns
                    for col in cat_cols:
                        # Replace NaN with mode
                        most_common = data[col].mode()[0] if not data[col].mode().empty else "Unknown"
                        data[col] = data[col].fillna(most_common)
                    
                    st.success("✅ Missing and infinity values have been fixed automatically.")
                
                # Update progress
                progress_bar.progress(20)
                time.sleep(0.3)  # Simulate processing time
                
                # Check if target column exists
                if 'target_column' not in st.session_state or st.session_state.target_column is None:
                    # Try to identify target column
                    potential_target = identify_target_column(data)
                    if potential_target:
                        st.session_state.target_column = potential_target
                        st.info(f"✅ Automatically identified '{potential_target}' as the target column.")
                    else:
                        st.error("❌ No target column identified. Please go to Data Overview and select a target column.")
                        return
                
                # Make sure target column exists in the data
                if st.session_state.target_column not in data.columns:
                    st.error(f"❌ Target column '{st.session_state.target_column}' not found in data.")
                    return
                
                # Update progress
                progress_bar.progress(40)
                time.sleep(0.3)  # Simulate processing time
                
                # Separate features and target
                X = data.drop(columns=[st.session_state.target_column]).values
                y = data[st.session_state.target_column].values
                feature_names = data.drop(columns=[st.session_state.target_column]).columns.tolist()
                
                X_engineered, engineered_feature_names = st.session_state.feature_engineer.engineer_features(X, feature_names)
                
                # Store engineered features information
                st.session_state.engineered_feature_count = len(engineered_feature_names) - len(feature_names)
                
                # Update progress
                progress_bar.progress(80)
                time.sleep(0.3)  # Simulate processing time
                
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
                    
                    # Final progress update
                    progress_bar.progress(100)
                    
                    st.success(f"✅ Feature engineering and selection completed! Selected {len(selected_feature_names)} features from {len(engineered_feature_names)} engineered features.")
                else:
                    # Create DataFrame with all engineered features
                    engineered_df = pd.DataFrame(X_engineered, columns=engineered_feature_names)
                    
                    # Add target column back
                    engineered_df[st.session_state.target_column] = y
                    
                    # Store in session state
                    st.session_state.features = engineered_df
                    st.session_state.selected_feature_names = engineered_feature_names
                    
                    # Final progress update
                    progress_bar.progress(100)
                    
                    st.success(f"✅ Feature engineering completed! Created {st.session_state.engineered_feature_count} new features.")
                
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
                        
                        # Convert to plot-friendly format
                        feature_importance = prepare_dataframe_for_plotting(feature_importance)
                        feature_importance = feature_importance.sort_values('Importance', ascending=False)
                        
                        st.subheader("Feature Importance")
                        
                        fig = px.bar(
                            feature_importance.head(20),
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title="Top 20 Features by Importance",
                            color='Importance',
                            color_continuous_scale='viridis'
                        )
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Navigate to next step
                st.markdown("""
                <div style='background-color: var(--background-secondary); padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
                    <h4 style='margin-top: 0;'>Next Step: Model Training</h4>
                    <p>Your data now has engineered features that can improve model performance. Go to the Model Training & Evaluation page to train machine learning models.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Preview final features
                st.subheader("Engineered Features Preview")
                st.dataframe(st.session_state.features.head(), use_container_width=True)
        
        except Exception as e:
            st.error(f"Error during feature engineering: {str(e)}")
            st.exception(e)  # This shows the full traceback for debugging
    
    # Display existing features if available
    if 'features' in st.session_state and st.session_state.features is not None:
        with st.expander("View Current Engineered Features", expanded=False):
            st.info(f"Engineered data shape: {st.session_state.features.shape[0]:,} rows × {st.session_state.features.shape[1]:,} columns")
            st.dataframe(st.session_state.features.head(), use_container_width=True)

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