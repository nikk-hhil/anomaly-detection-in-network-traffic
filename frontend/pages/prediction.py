import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import time
from typing import Dict, List, Any, Tuple
from datetime import datetime
from frontend.utils.visualization import prepare_dataframe_for_plotting

# Add the project root to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from src modules
from src.anomaly_detector import AnomalyDetector

def show_prediction_page():
    """Display the anomaly detection and prediction page."""
    st.title("🔍 Anomaly Detection")
    
    # Check if a model has been trained and selected for prediction
    if 'prediction_model' not in st.session_state or st.session_state.prediction_model is None:
        if 'best_model' in st.session_state and st.session_state.best_model is not None:
            # Use best model if available
            st.session_state.prediction_model = st.session_state.best_model['model']
            st.session_state.prediction_model_name = st.session_state.best_model['name']
            st.info(f"Using the best model ({st.session_state.best_model['name'].title().replace('_', ' ')}) for prediction.")
        else:
            st.warning("No model available for prediction. Please train models in the Training tab first.")
            st.info("Go to the **Model Training & Evaluation** page to train and select a model.")
            return
    
    # Check if preprocessor and feature engineer are available
    if 'preprocessor' not in st.session_state or 'feature_engineer' not in st.session_state:
        st.warning("Preprocessor or feature engineer not found. Please complete data preprocessing and feature engineering first.")
        st.info("Go to the **Data Upload & Exploration** page to prepare your data.")
        return
    
    # Create tabs for the workflow
    tabs = st.tabs(["Data Upload", "Anomaly Detection", "Results Visualization", "Explanation"])
    
    # Tab 1: Data Upload
    with tabs[0]:
        show_data_upload_tab()
    
    # Tab 2: Anomaly Detection
    with tabs[1]:
        show_anomaly_detection_tab()
    
    # Tab 3: Results Visualization
    with tabs[2]:
        show_results_visualization_tab()
    
    # Tab 4: Explanation
    with tabs[3]:
        show_explanation_tab()

def show_data_upload_tab():
    """Show data upload functionality for prediction."""
    st.header("Upload New Network Traffic Data")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file with network traffic data for prediction", type="csv")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sample dataset option
        st.markdown("### Or use a sample test dataset")
        use_sample = st.button("Load Sample Test Dataset")
        
        if use_sample:
            try:
                # Check if sample data exists
                sample_path = "data/sample_test_traffic.csv"
                if os.path.exists(sample_path):
                    with st.spinner("Loading sample test dataset..."):
                        data = pd.read_csv(sample_path)
                        st.session_state.prediction_data = data
                        st.success(f"Loaded sample test dataset with {data.shape[0]} rows and {data.shape[1]} columns!")
                else:
                    st.error("Sample test dataset not found. Please upload your own data.")
            except Exception as e:
                st.error(f"Error loading sample test dataset: {str(e)}")
    
    with col2:
        # Dataset information
        st.markdown("### Dataset Requirements")
        st.markdown("""
        The dataset for prediction should:
        - Be in CSV format
        - Contain the same features as your training data
        - May or may not contain the target/label column
        
        The system will preprocess this data using the same steps applied to the training data.
        """)
    
    # Process the uploaded file
    if uploaded_file is not None:
        try:
            with st.spinner("Loading and processing data..."):
                # Load the data
                data = pd.read_csv(uploaded_file)
                
                for col in data.columns:
                    if str(data[col].dtype).startswith(('Int', 'Float')):
                        data[col] = data[col].astype(float)
                
                # Store in session state
                st.session_state.prediction_data = data
                
                # Reset prediction results
                st.session_state.prediction_results = None
                
                # Success message
                st.success(f"Successfully loaded data with {data.shape[0]} rows and {data.shape[1]} columns!")
                
                # Show quick preview
                st.subheader("Data Preview")
                st.dataframe(data.head())
        
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    
    # Display existing data if available
    if 'prediction_data' in st.session_state and st.session_state.prediction_data is not None:
        st.subheader("Current Prediction Data")
        st.info(f"Data shape: {st.session_state.prediction_data.shape}")
        st.dataframe(st.session_state.prediction_data.head())

def show_anomaly_detection_tab():
    """Show anomaly detection controls and run prediction."""
    st.header("Detect Anomalies")
    
    # Check if prediction data is available
    if 'prediction_data' not in st.session_state or st.session_state.prediction_data is None:
        st.warning("Please upload data for prediction in the Data Upload tab.")
        return
    
    # Model info
    st.subheader("Model Information")
    
    model_name = st.session_state.prediction_model_name.title().replace('_', ' ')
    st.write(f"Using model: **{model_name}**")
    
    # Check if target column exists in prediction data
    target_in_data = False
    if 'target_column' in st.session_state and st.session_state.target_column in st.session_state.prediction_data.columns:
        target_in_data = True
        st.info(f"Target column '{st.session_state.target_column}' found in data. Evaluation metrics will be calculated.")
    
    # Prediction options
    st.subheader("Detection Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Option to adjust threshold for binary classification
        if hasattr(st.session_state.prediction_model, 'predict_proba'):
            threshold = st.slider(
                "Detection Threshold",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.05,
                help="Probability threshold for classifying an instance as an anomaly"
            )
        else:
            threshold = 0.5
            st.info("This model doesn't support probability predictions, so threshold adjustment is not available.")
    
    with col2:
        # Option to process in batches for large datasets
        batch_processing = st.checkbox(
            "Use Batch Processing",
            value=st.session_state.prediction_data.shape[0] > 10000,
            help="Process large datasets in batches to reduce memory usage"
        )
        
        if batch_processing:
            batch_size = st.number_input(
                "Batch Size",
                min_value=1000,
                max_value=100000,
                value=10000,
                step=1000,
                help="Number of rows to process in each batch"
            )
    
    # Run prediction button
    if st.button("Detect Anomalies", use_container_width=True):
        try:
            with st.spinner("Detecting anomalies... This may take some time for large datasets."):
                # Initialize AnomalyDetector
                detector = initialize_anomaly_detector(threshold)
                
                # Select the target column if it exists in the data
                target_column = st.session_state.target_column if target_in_data else None
                
                # Get the prediction data
                data = st.session_state.prediction_data
                
                # Make predictions
                if batch_processing and data.shape[0] > batch_size:
                    # Use batch prediction
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = os.path.join("results", f"batch_predictions_{timestamp}.csv")
                    
                    # Ensure output directory exists
                    os.makedirs("results", exist_ok=True)
                    
                    # Run batch prediction
                    detector.predict_batch(
                        input_file=create_temp_csv(data),
                        output_file=output_path,
                        target_column=target_column,
                        batch_size=batch_size
                    )
                    
                    # Load the results
                    results = pd.read_csv(output_path)
                    
                    st.session_state.prediction_results = results
                    st.session_state.prediction_output_path = output_path
                    
                    st.success(f"Successfully processed {data.shape[0]} rows in batches.")
                    st.info(f"Results saved to {output_path}")
                else:
                    # Standard prediction
                    y_pred, y_proba = detector.predict(data, target_column)
                    
                    # Create result dataframe
                    results = data.copy()
                    results['predicted_class'] = y_pred
                    
                    # Add class names if available
                    if detector.label_mapping:
                        results['predicted_label'] = [detector._get_class_name(pred) for pred in y_pred]
                    
                    # Add probabilities if available
                    if y_proba is not None:
                        if y_proba.ndim > 1:
                            # Multi-class case
                            for j in range(y_proba.shape[1]):
                                results[f'probability_class_{j}'] = y_proba[:, j]
                                if detector.label_mapping:
                                    class_name = detector._get_class_name(j)
                                    results[f'probability_{class_name}'] = y_proba[:, j]
                        else:
                            # Binary case
                            results['probability'] = y_proba
                    
                    # Store results
                    st.session_state.prediction_results = results
                    
                    # Evaluate if target column is available
                    if target_in_data:
                        metrics = detector.evaluate(data, target_column)
                        st.session_state.prediction_metrics = metrics
                        
                        # Display metric summary
                        st.subheader("Evaluation Metrics")
                        
                        accuracy = metrics.get('accuracy', 0)
                        precision = metrics.get('precision', 0)
                        recall = metrics.get('recall', 0)
                        f1 = metrics.get('f1_score', 0)
                        
                        st.write(f"Accuracy: {accuracy:.4f}")
                        st.write(f"Precision: {precision:.4f}")
                        st.write(f"Recall: {recall:.4f}")
                        st.write(f"F1 Score: {f1:.4f}")
                    
                    st.success(f"Successfully detected anomalies in {data.shape[0]} records!")
                
                # Prompt to go to visualization tab
                st.info("Go to the Results Visualization tab to explore the detection results.")
        
        except Exception as e:
            st.error(f"Error during anomaly detection: {str(e)}")
    
    # Display existing results if available
    if 'prediction_results' in st.session_state and st.session_state.prediction_results is not None:
        st.subheader("Prediction Results Summary")
        
        results = st.session_state.prediction_results
        
        # Check if we have predicted labels
        if 'predicted_label' in results.columns:
            # Show predicted class distribution
            pred_counts = results['predicted_label'].value_counts().reset_index()
            pred_counts.columns = ['Class', 'Count']
            pred_counts['Percentage'] = 100 * pred_counts['Count'] / pred_counts['Count'].sum()
            
            fig = px.pie(
                pred_counts,
                values='Count',
                names='Class',
                title='Distribution of Predicted Classes',
                hole=0.4,
                hover_data=['Percentage']
            )
            
            st.plotly_chart(fig, use_container_width=True)
        elif 'predicted_class' in results.columns:
            # Show predicted class distribution
            pred_counts = results['predicted_class'].value_counts().reset_index()
            pred_counts.columns = ['Class', 'Count']
            pred_counts['Percentage'] = 100 * pred_counts['Count'] / pred_counts['Count'].sum()
            
            fig = px.pie(
                pred_counts,
                values='Count',
                names='Class',
                title='Distribution of Predicted Classes',
                hole=0.4,
                hover_data=['Percentage']
            )
            
            st.plotly_chart(fig, use_container_width=True)

def initialize_anomaly_detector(threshold: float = 0.5) -> AnomalyDetector:
    """Initialize the AnomalyDetector with required components."""
    # Save the model to a temporary file
    import tempfile
    import joblib
    
    # Create temp directory
    os.makedirs("temp", exist_ok=True)
    
    # Save model
    model_path = f"temp/model_{int(time.time())}.joblib"
    joblib.dump(st.session_state.prediction_model, model_path)
    
    # Save preprocessor
    preprocessor_path = f"temp/preprocessor_{int(time.time())}.joblib"
    joblib.dump(st.session_state.preprocessor, preprocessor_path)
    
    # Save feature engineer
    feature_engineer_path = f"temp/feature_engineer_{int(time.time())}.joblib"
    joblib.dump(st.session_state.feature_engineer, feature_engineer_path)
    
    # Create output directory
    os.makedirs("results", exist_ok=True)
    
    # Initialize detector
    detector = AnomalyDetector(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        feature_engineer_path=feature_engineer_path,
        threshold=threshold,
        output_dir="results"
    )
    
    return detector

def create_temp_csv(data: pd.DataFrame) -> str:
    """Create a temporary CSV file for batch processing."""
    # Create temp directory
    os.makedirs("temp", exist_ok=True)
    
    # Create unique file name
    temp_file = f"temp/batch_input_{int(time.time())}.csv"
    
    # Save to CSV
    data.to_csv(temp_file, index=False)
    
    return temp_file

def show_results_visualization_tab():
    """Show visualization of prediction results."""
    st.header("Results Visualization")
    
    # Check if prediction results are available
    if 'prediction_results' not in st.session_state or st.session_state.prediction_results is None:
        st.warning("No prediction results available. Please run anomaly detection first.")
        return
    
    # Get results
    results = st.session_state.prediction_results
    
    # Create visualization options
    st.subheader("Select Visualization")
    
    viz_type = st.radio(
        "Visualization Type",
        ["Predicted Class Distribution", "Feature Analysis by Class", "Anomaly Timeline", "Confidence Distribution"]
    )
    
    # Get class column
    class_column = 'predicted_label' if 'predicted_label' in results.columns else 'predicted_class'
    
    # Check if we have class probabilities
    has_probabilities = any(col.startswith('probability_') for col in results.columns)
    
    # Show selected visualization
    if viz_type == "Predicted Class Distribution":
        show_class_distribution(results, class_column)
    
    elif viz_type == "Feature Analysis by Class":
        show_feature_analysis_by_class(results, class_column)
    
    elif viz_type == "Anomaly Timeline":
        show_anomaly_timeline(results, class_column)
    
    elif viz_type == "Confidence Distribution" and has_probabilities:
        show_confidence_distribution(results, class_column)
    
    elif viz_type == "Confidence Distribution":
        st.info("Confidence visualization requires probability outputs, which are not available for this model.")
    
    # Export options
    st.subheader("Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Download CSV Results"):
            # Create a download link
            csv = results.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"anomaly_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("View Full Results"):
            st.subheader("Complete Results Data")
            st.dataframe(results, use_container_width=True)

def show_class_distribution(results: pd.DataFrame, class_column: str):
    """Show distribution of predicted classes."""
    st.subheader("Distribution of Predicted Classes")
    
    # Get class counts
    class_counts = results[class_column].value_counts().reset_index()
    class_counts.columns = ['Class', 'Count']
    class_counts['Percentage'] = 100 * class_counts['Count'] / class_counts['Count'].sum()
    
    class_counts = prepare_dataframe_for_plotting(class_counts)
    
    # Bar chart
    fig = px.bar(
        class_counts,
        x='Class',
        y='Count',
        color='Class',
        text=class_counts['Percentage'].apply(lambda x: f'{x:.1f}%'),
        title='Distribution of Predicted Classes'
    )
    
    fig.update_layout(xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Display table with counts
    st.subheader("Class Distribution Details")
    st.dataframe(class_counts, use_container_width=True)
    
    # Anomaly summary
    if len(class_counts) > 1:  # Multiple classes
        normal_class = None
        
        # Try to identify normal traffic class (either labeled 'normal' or has highest count)
        if 'normal' in results[class_column].values:
            normal_class = 'normal'
        elif 'Normal' in results[class_column].values:
            normal_class = 'Normal'
        elif 'BENIGN' in results[class_column].values:
            normal_class = 'BENIGN'
        else:
            # Assume the most common class is normal traffic
            normal_class = class_counts.iloc[0]['Class']
        
        # Calculate anomaly percentage
        if normal_class:
            normal_count = results[results[class_column] == normal_class].shape[0]
            anomaly_count = results[results[class_column] != normal_class].shape[0]
            anomaly_percent = 100 * anomaly_count / (normal_count + anomaly_count)
            
            st.subheader("Anomaly Summary")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Records", f"{results.shape[0]:,}")
            col2.metric("Normal Traffic", f"{normal_count:,}")
            col3.metric("Anomalies Detected", f"{anomaly_count:,} ({anomaly_percent:.1f}%)")
            
            # List attack types
            if anomaly_count > 0:
                st.subheader("Attack Types Detected")
                
                attack_counts = results[results[class_column] != normal_class][class_column].value_counts().reset_index()
                attack_counts.columns = ['Attack Type', 'Count']
                attack_counts['Percentage'] = 100 * attack_counts['Count'] / attack_counts['Count'].sum()
                
                fig = px.pie(
                    attack_counts,
                    values='Count',
                    names='Attack Type',
                    title='Distribution of Attack Types',
                    hole=0.4
                )
                
                st.plotly_chart(fig, use_container_width=True)

def show_feature_analysis_by_class(results: pd.DataFrame, class_column: str):
    """Show feature analysis grouped by predicted class."""
    st.subheader("Feature Analysis by Class")
    
    # Get numeric columns for analysis
    numeric_cols = results.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove prediction columns
    numeric_cols = [col for col in numeric_cols if not col.startswith('predicted_') and not col.startswith('probability_')]
    
    if not numeric_cols:
        st.warning("No numerical features found for analysis.")
        return
    
    # Select features to analyze
    selected_features = st.multiselect(
        "Select features to analyze",
        options=numeric_cols,
        default=numeric_cols[:3] if len(numeric_cols) > 3 else numeric_cols
    )
    
    if not selected_features:
        st.info("Please select at least one feature to analyze.")
        return
    
    # Get unique classes
    classes = results[class_column].unique()
    
    # Select visualization type
    viz_type = st.selectbox(
        "Select visualization type",
        ["Box Plot", "Violin Plot", "Histogram", "Scatter Plot"]
    )
    
    plot_data = prepare_dataframe_for_plotting(results)
    
    # Show selected visualization
    if viz_type == "Box Plot":
        for feature in selected_features:
            fig = px.box(
                plot_data,
                x=class_column,
                y=feature,
                color=class_column,
                title=f"Distribution of {feature} by Class",
                notched=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Violin Plot":
        for feature in selected_features:
            fig = px.violin(
                results,
                x=class_column,
                y=feature,
                color=class_column,
                title=f"Distribution of {feature} by Class",
                box=True,
                points="all"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Histogram":
        for feature in selected_features:
            fig = px.histogram(
                results,
                x=feature,
                color=class_column,
                marginal="box",
                title=f"Distribution of {feature} by Class",
                barmode="overlay",
                opacity=0.7
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Scatter Plot":
        if len(selected_features) < 2:
            st.warning("Please select at least two features for scatter plot.")
            return
        
        # Select features for X and Y axes
        x_feature = st.selectbox("Select X-axis feature", options=selected_features, index=0)
        y_feature = st.selectbox("Select Y-axis feature", options=selected_features, index=min(1, len(selected_features)-1))
        
        fig = px.scatter(
            results,
            x=x_feature,
            y=y_feature,
            color=class_column,
            title=f"Scatter Plot: {x_feature} vs {y_feature} by Class",
            opacity=0.7
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_anomaly_timeline(results: pd.DataFrame, class_column: str):
    """Show timeline of anomalies if time-related columns exist."""
    st.subheader("Anomaly Timeline")
    
    # Look for time-related columns
    time_cols = [col for col in results.columns if any(term in col.lower() for term in ['time', 'date', 'timestamp', 'hour', 'minute', 'second'])]
    
    if not time_cols:
        st.info("No time-related columns found in the dataset. Cannot create timeline visualization.")
        return
    
    # Select time column
    time_column = st.selectbox("Select time column", options=time_cols)
    
    # Try to parse time column
    try:
        # Add a copy of the column with parsed timestamps
        results['_parsed_time'] = pd.to_datetime(results[time_column], errors='coerce')
        
        # Check if parsing was successful
        if results['_parsed_time'].isna().all():
            st.warning(f"Could not parse column '{time_column}' as timestamps.")
            return
        
        # Remove rows with missing timestamps
        valid_results = results.dropna(subset=['_parsed_time']).copy()
        
        # Create timeline visualization
        st.subheader(f"Anomaly Timeline using {time_column}")
        
        # Count anomalies by time period
        if valid_results['_parsed_time'].nunique() > 50:
            # Too many unique timestamps, group by hour
            valid_results['_time_group'] = valid_results['_parsed_time'].dt.floor('H')
            time_group_name = "Hour"
        elif valid_results['_parsed_time'].nunique() > 20:
            # Group by 10-minute intervals
            valid_results['_time_group'] = valid_results['_parsed_time'].dt.floor('10min')
            time_group_name = "10-Minute Interval"
        else:
            # Use original timestamps
            valid_results['_time_group'] = valid_results['_parsed_time']
            time_group_name = "Timestamp"
        
        # Count occurrences by time group and class
        time_counts = valid_results.groupby(['_time_group', class_column]).size().reset_index(name='Count')
        
        time_counts = prepare_dataframe_for_plotting(time_counts)
        
        # Create timeline plot
        fig = px.line(
            time_counts,
            x='_time_group',
            y='Count',
            color=class_column,
            markers=True,
            title=f"Anomaly Timeline by {time_group_name}",
            labels={'_time_group': time_group_name, 'Count': 'Number of Records'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create heatmap of anomalies over time
        st.subheader("Anomaly Heatmap")
        
        # Pivot data for heatmap
        heatmap_data = time_counts.pivot(index='_time_group', columns=class_column, values='Count').fillna(0)
        
        # Create heatmap
        fig = px.imshow(
            heatmap_data.T,
            labels=dict(x=time_group_name, y="Class", color="Count"),
            title=f"Anomaly Heatmap by {time_group_name}",
            color_continuous_scale="Viridis"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating timeline visualization: {str(e)}")

def show_confidence_distribution(results: pd.DataFrame, class_column: str):
    """Show distribution of prediction confidence."""
    st.subheader("Prediction Confidence Distribution")
    
    # Get probability columns
    prob_cols = [col for col in results.columns if col.startswith('probability_')]
    
    if not prob_cols:
        st.info("No probability columns found. This visualization requires probability outputs.")
        return
    
    # If we have class-specific probabilities
    if len(prob_cols) > 1 and not any(col == 'probability' for col in prob_cols):
        # Calculate confidence as the maximum probability
        results['confidence'] = results[prob_cols].max(axis=1)
        
        # Create histogram of confidence by class
        fig = px.histogram(
            results,
            x='confidence',
            color=class_column,
            nbins=30,
            marginal="box",
            title="Distribution of Prediction Confidence by Class",
            opacity=0.7
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create confusion matrix-like heatmap of confidence
        st.subheader("Confidence by Class")
        
        # Calculate average confidence for each class
        confidence_by_class = results.groupby(class_column)['confidence'].mean().reset_index()
        
        fig = px.bar(
            confidence_by_class,
            x=class_column,
            y='confidence',
            color=class_column,
            title="Average Confidence by Class",
            labels={'confidence': 'Average Confidence'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # For binary classification with single probability column
    elif 'probability' in results.columns:
        # Create histogram of probability
        fig = px.histogram(
            results,
            x='probability',
            color=class_column,
            nbins=30,
            marginal="box",
            title="Distribution of Prediction Probability",
            opacity=0.7
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create threshold analysis
        st.subheader("Threshold Analysis")
        
        # Get thresholds to evaluate
        thresholds = np.linspace(0.1, 0.9, 9)
        
        # Calculate class distribution at different thresholds
        threshold_results = []
        
        for threshold in thresholds:
            # Apply threshold
            predicted_class = (results['probability'] >= threshold).astype(int)
            
            # Count classes
            class_counts = pd.Series(predicted_class).value_counts()
            
            # Get counts (handling missing classes)
            class_0_count = class_counts.get(0, 0)
            class_1_count = class_counts.get(1, 0)
            
            # Calculate percentages
            total = class_0_count + class_1_count
            
            threshold_results.append({
                'Threshold': threshold,
                'Class 0 Count': class_0_count,
                'Class 1 Count': class_1_count,
                'Class 0 Percentage': 100 * class_0_count / total,
                'Class 1 Percentage': 100 * class_1_count / total
            })
        
        # Create dataframe
        threshold_df = pd.DataFrame(threshold_results)
        
        # Create area chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=threshold_df['Threshold'],
            y=threshold_df['Class 0 Percentage'],
            name='Class 0',
            mode='lines',
            line=dict(width=0),
            stackgroup='one',
            groupnorm='percent'
        ))
        
        fig.add_trace(go.Scatter(
            x=threshold_df['Threshold'],
            y=threshold_df['Class 1 Percentage'],
            name='Class 1',
            mode='lines',
            line=dict(width=0),
            stackgroup='one'
        ))
        
        fig.update_layout(
            title="Class Distribution by Threshold",
            xaxis_title="Threshold",
            yaxis_title="Percentage",
            legend_title="Class",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
# Show threshold table
        st.subheader("Threshold Details")
        st.dataframe(threshold_df, use_container_width=True)

def show_explanation_tab():
    """Show explanation of predictions for specific examples."""
    st.header("Prediction Explanation")
    
    # Check if prediction results are available
    if 'prediction_results' not in st.session_state or st.session_state.prediction_results is None:
        st.warning("No prediction results available. Please run anomaly detection first.")
        return
    
    # Get results
    results = st.session_state.prediction_results
    
    # Get class column
    class_column = 'predicted_label' if 'predicted_label' in results.columns else 'predicted_class'
    
    # Select specific examples for explanation
    st.subheader("Select Examples to Explain")
    
    # Get unique classes
    classes = results[class_column].unique()
    
    # Select examples by class
    selected_class = st.selectbox(
        "Select class to explain",
        options=classes
    )
    
    # Get examples of selected class
    class_examples = results[results[class_column] == selected_class]
    
    if len(class_examples) == 0:
        st.warning(f"No examples found for class '{selected_class}'.")
        return
    
    # Select specific example
    if len(class_examples) > 100:
        # If there are many examples, use a random sample
        sample_indices = class_examples.sample(n=100).index.tolist()
        example_idx = st.selectbox(
            "Select example to explain",
            options=sample_indices,
            format_func=lambda x: f"Example {x} (Class: {selected_class})"
        )
    else:
        # Otherwise, show all examples
        example_idx = st.selectbox(
            "Select example to explain",
            options=class_examples.index.tolist(),
            format_func=lambda x: f"Example {x} (Class: {selected_class})"
        )
    
    # Get selected example
    selected_example = results.loc[example_idx]
    
    # Display the example
    st.subheader("Selected Example Details")
    
    # Display top-level metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Index:** {example_idx}")
        st.markdown(f"**Predicted Class:** {selected_example[class_column]}")
        
        # Display confidence if available
        prob_cols = [col for col in selected_example.index if col.startswith('probability_')]
        if prob_cols:
            if 'probability' in selected_example:
                # Binary case
                st.markdown(f"**Confidence:** {selected_example['probability']:.4f}")
            else:
                # Multi-class case - get probability for predicted class
                for prob_col in prob_cols:
                    if prob_col.endswith(f"_{selected_class}") or prob_col.endswith(f"_class_{selected_example['predicted_class']}"):
                        st.markdown(f"**Confidence:** {selected_example[prob_col]:.4f}")
                        break
    
    with col2:
        # Display target class if available
        if 'target_column' in st.session_state and st.session_state.target_column in results.columns:
            target_column = st.session_state.target_column
            st.markdown(f"**Actual Class:** {selected_example[target_column]}")
            
            # Check if prediction was correct
            is_correct = selected_example[target_column] == selected_example[class_column]
            st.markdown(f"**Prediction:** {'✅ Correct' if is_correct else '❌ Incorrect'}")
    
    # Show all features in a table
    st.subheader("Feature Values")
    
    # Get feature values (excluding prediction columns)
    feature_cols = [col for col in selected_example.index 
                   if not col.startswith('predicted_') and not col.startswith('probability_')]
    
    # Create a DataFrame for display
    feature_df = pd.DataFrame({
        'Feature': feature_cols,
        'Value': [selected_example[col] for col in feature_cols]
    })
    
    st.dataframe(feature_df, use_container_width=True)
    
    # Try to explain the prediction
    try:
        # Check if we can initialize an AnomalyDetector
        if 'preprocessor' in st.session_state and 'feature_engineer' in st.session_state and 'prediction_model' in st.session_state:
            st.subheader("Feature Importance for this Prediction")
            
            # Initialize detector
            detector = initialize_anomaly_detector()
            
            # Get explanation
            explanation = None
            
            try:
                # Attempt to get explanation
                explanation = detector.explain_prediction(
                    results, 
                    example_idx, 
                    target_column=st.session_state.target_column if 'target_column' in st.session_state else None
                )
            except Exception as e:
                st.warning(f"Could not generate detailed explanation: {str(e)}")
            
            if explanation is not None and 'feature_importances' in explanation:
                # Create a dataframe of feature importances
                importances = pd.DataFrame(explanation['feature_importances'])
                
                # Sort by importance
                importances = importances.sort_values('importance', ascending=False)
                
                # Show bar chart
                fig = px.bar(
                    importances.head(20),
                    x='importance',
                    y='name',
                    orientation='h',
                    title='Top 20 Features Contributing to this Prediction',
                    color='importance',
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Decision breakdown
                st.subheader("Decision Breakdown")
                
                st.markdown("""
                The features shown above had the strongest influence on the model's decision.
                Positive values indicate features pushing toward the predicted class,
                while negative values (if any) indicate features pushing against it.
                """)
                
                if 'confidence' in explanation['prediction']:
                    st.markdown(f"The model made this prediction with a confidence of **{explanation['prediction']['confidence']:.2f}**.")
            else:
                st.info("Detailed feature importance explanation is not available for this model type or example.")
        else:
            st.info("Model components not available for detailed explanation.")
    
    except Exception as e:
        st.error(f"Error generating explanation: {str(e)}")
    
    # Compare with similar examples
    st.subheader("Similar Examples")
    
    # Get examples of the same class
    similar_examples = results[results[class_column] == selected_class].drop(example_idx)
    
    if len(similar_examples) == 0:
        st.info("No other examples of this class found.")
        return
    
    # Sample a few examples (up to 5)
    num_similar = min(5, len(similar_examples))
    similar_sample = similar_examples.sample(n=num_similar)
    
    # Create a dataframe with feature values
    # First, get numeric features
    numeric_features = results.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_features = [f for f in numeric_features if not f.startswith('predicted_') and not f.startswith('probability_')]
    
    # Limit to top 10 features
    if len(numeric_features) > 10:
        numeric_features = numeric_features[:10]
    
    # Create comparison dataframe
    comparison_data = []
    
    # Selected example
    selected_example_row = {'Example': f'Selected ({example_idx})'}
    for feature in numeric_features:
        selected_example_row[feature] = selected_example[feature]
    comparison_data.append(selected_example_row)
    
    # Similar examples
    for i, (idx, example) in enumerate(similar_sample.iterrows()):
        example_row = {'Example': f'Similar {i+1} ({idx})'}
        for feature in numeric_features:
            example_row[feature] = example[feature]
        comparison_data.append(example_row)
    
    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display comparison table
    st.dataframe(comparison_df, use_container_width=True)
    
    # Calculate and display distance to similar examples
    if len(numeric_features) > 0:
        st.subheader("Distance to Similar Examples")
        
        # Create a copy of results with only numeric features
        numeric_data = results[numeric_features].copy()
        
        # Standardize features for distance calculation
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        numeric_data_scaled = pd.DataFrame(
            scaler.fit_transform(numeric_data),
            columns=numeric_data.columns,
            index=numeric_data.index
        )
        
        # Calculate Euclidean distance from selected example to all others
        example_vector = numeric_data_scaled.loc[example_idx].values
        
        distances = []
        for idx, row in numeric_data_scaled.iterrows():
            if idx != example_idx:
                # Calculate Euclidean distance
                distance = np.sqrt(np.sum((example_vector - row.values) ** 2))
                distances.append((idx, distance))
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        
        # Take the top 20 closest examples
        closest_examples = distances[:20]
        
        # Create a DataFrame for visualization
        closest_df = pd.DataFrame(closest_examples, columns=['Index', 'Distance'])
        closest_df['Class'] = [results.loc[idx, class_column] for idx in closest_df['Index']]
        
        # Create bar chart of distances
        fig = px.bar(
            closest_df,
            x='Distance',
            y='Index',
            color='Class',
            title='20 Closest Examples (Euclidean Distance)',
            orientation='h'
        )
        
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Show a summary of similar classes
        similar_class_counts = closest_df['Class'].value_counts().reset_index()
        similar_class_counts.columns = ['Class', 'Count']
        
        fig = px.pie(
            similar_class_counts,
            values='Count',
            names='Class',
            title='Class Distribution of Similar Examples',
            hole=0.4
        )
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    # For testing the page in isolation
    import streamlit as st
    
    # Initialize session state
    if 'prediction_model' not in st.session_state:
        # Use a dummy model for testing
        from sklearn.ensemble import RandomForestClassifier
        st.session_state.prediction_model = RandomForestClassifier(n_estimators=10, random_state=42)
        st.session_state.prediction_model_name = "random_forest"
    
    if 'preprocessor' not in st.session_state:
        from src.preprocessor import DataPreprocessor
        st.session_state.preprocessor = DataPreprocessor()
    
    if 'feature_engineer' not in st.session_state:
        from src.feature_engineering import FeatureEngineer
        st.session_state.feature_engineer = FeatureEngineer()
    
    # Display the page
    show_prediction_page()