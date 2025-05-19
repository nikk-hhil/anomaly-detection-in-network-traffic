import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add the project root to the path so we can import from src if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def show_home_page():
    """Display the home page with overview and key metrics."""
    # Page title and introduction with better styling
    st.markdown("<h1 style='text-align: center;'>Network Traffic Anomaly Detection Dashboard</h1>", unsafe_allow_html=True)
    
    # Dashboard overview with improved layout
    st.markdown("""
    <div style='text-align: center; padding: 1rem; margin-bottom: 2rem;'>
        This system uses machine learning to identify and classify network traffic anomalies
        and potential cyber attacks by analyzing network flow features.
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics section
    display_key_metrics()
    
    # Feature cards section
    display_feature_cards()
    
    # Show workflow guide
    display_workflow_guide()
    
    # Show attack types information
    st.markdown("<hr>", unsafe_allow_html=True)
    show_attack_types_info()

def display_key_metrics():
    """Display key metrics with improved cards."""
    st.markdown("<h2>System Overview</h2>", unsafe_allow_html=True)
    
    # Create 4 metric cards in a row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='dashboard-card'>
            <h3 style='text-align: center; margin-top: 0;'>Data Status</h3>
            <div style='text-align: center; font-size: 3rem; color: #1e3a5c;'>
                {}
            </div>
            <div style='text-align: center;'>
                {}
            </div>
        </div>
        """.format(
            "✅" if st.session_state.data is not None else "❌",
            f"{st.session_state.data.shape[0]:,} records" if st.session_state.data is not None else "No data loaded"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='dashboard-card'>
            <h3 style='text-align: center; margin-top: 0;'>Features</h3>
            <div style='text-align: center; font-size: 3rem; color: #1e3a5c;'>
                {}
            </div>
            <div style='text-align: center;'>
                {}
            </div>
        </div>
        """.format(
            "✅" if st.session_state.features is not None else "❌",
            f"{st.session_state.features.shape[1]-1:,} features" if st.session_state.features is not None else "No features engineered"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='dashboard-card'>
            <h3 style='text-align: center; margin-top: 0;'>Models</h3>
            <div style='text-align: center; font-size: 3rem; color: #1e3a5c;'>
                {}
            </div>
            <div style='text-align: center;'>
                {}
            </div>
        </div>
        """.format(
            "✅" if st.session_state.trained_models else "❌",
            f"{len(st.session_state.trained_models)} models trained" if st.session_state.trained_models else "No models trained"
        ), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='dashboard-card'>
            <h3 style='text-align: center; margin-top: 0;'>Anomalies</h3>
            <div style='text-align: center; font-size: 3rem; color: #1e3a5c;'>
                {}
            </div>
            <div style='text-align: center;'>
                {}
            </div>
        </div>
        """.format(
            "✅" if 'prediction_results' in st.session_state and st.session_state.prediction_results is not None else "❌",
            "Detection complete" if 'prediction_results' in st.session_state and st.session_state.prediction_results is not None else "No anomalies detected"
        ), unsafe_allow_html=True)
    
    # Display data visualization if data is available
    if st.session_state.data is not None and st.session_state.target_column:
        st.markdown("<br>", unsafe_allow_html=True)
        display_data_visualization()

def display_data_visualization():
    """Display visualizations if data is available."""
    # Create tabs for different visualizations
    viz_tabs = st.tabs(["Class Distribution", "Feature Analysis", "Model Performance"])
    
    with viz_tabs[0]:
        # Show class distribution if target column is identified
        if st.session_state.target_column:
            st.markdown("<h3>Traffic Class Distribution</h3>", unsafe_allow_html=True)
            
            data = st.session_state.data
            target_col = st.session_state.target_column
            
            class_counts = data[target_col].value_counts()
            
            # Create a pie chart of the classes
            fig = px.pie(
                values=class_counts.values,
                names=class_counts.index,
                title="Distribution of Network Traffic Classes",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[1]:
        # Show feature importance if available
        if 'feature_engineer' in st.session_state and st.session_state.features is not None:
            st.markdown("<h3>Top Features by Importance</h3>", unsafe_allow_html=True)
            
            # Show dummy feature importance chart
            if 'best_model' in st.session_state and st.session_state.best_model is not None:
                if hasattr(st.session_state.best_model['model'], 'feature_importances_'):
                    # Get feature importances
                    features = st.session_state.features.drop(columns=[st.session_state.target_column]).columns
                    importances = st.session_state.best_model['model'].feature_importances_
                    
                    # Create a dataframe for visualization
                    importance_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': importances
                    })
                    importance_df = importance_df.sort_values('Importance', ascending=False).head(10)
                    
                    # Create horizontal bar chart
                    fig = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Top 10 Features by Importance",
                        color='Importance',
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Feature importance visualization is not available for the current model type.")
            else:
                st.info("Train a model to see feature importance visualization.")
    
    with viz_tabs[2]:
        # Show model performance metrics if available
        if 'evaluation_results' in st.session_state and st.session_state.evaluation_results:
            st.markdown("<h3>Model Performance Comparison</h3>", unsafe_allow_html=True)
            
            # Create radar chart for model comparison
            models = list(st.session_state.evaluation_results.keys())
            metrics = st.session_state.evaluation_results
            
            # Create radar chart for model comparison
            categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            
            fig = go.Figure()
            
            for model in models:
                if model in metrics:
                    fig.add_trace(go.Scatterpolar(
                        r=[
                            metrics[model].get('accuracy', 0),
                            metrics[model].get('precision', 0),
                            metrics[model].get('recall', 0),
                            metrics[model].get('f1', 0)
                        ],
                        theta=categories,
                        fill='toself',
                        name=model.title().replace('_', ' ')
                    ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Train and evaluate models to see performance metrics.")

def display_feature_cards():
    """Display feature cards for the system capabilities."""
    st.markdown("<h2>Key Features</h2>", unsafe_allow_html=True)
    
    # Create two rows of feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='dashboard-card'>
            <h3 style='text-align: center;'>🔍 Advanced Detection</h3>
            <p>Identify multiple types of network anomalies and attacks using machine learning algorithms.</p>
            <ul>
                <li>DoS/DDoS Attack Detection</li>
                <li>Port Scanning Detection</li>
                <li>Data Exfiltration Detection</li>
                <li>Botnet Communication</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='dashboard-card'>
            <h3 style='text-align: center;'>⚙️ Customizable Models</h3>
            <p>Train and optimize multiple machine learning models to fit your network traffic patterns.</p>
            <ul>
                <li>Random Forest</li>
                <li>Gradient Boosting</li>
                <li>Support Vector Machines</li>
                <li>Ensemble Models</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='dashboard-card'>
            <h3 style='text-align: center;'>📊 Interactive Visualization</h3>
            <p>Explore and analyze network traffic data and detection results with interactive visualizations.</p>
            <ul>
                <li>Traffic Pattern Analysis</li>
                <li>Anomaly Timeline</li>
                <li>Feature Importance</li>
                <li>Model Performance Comparison</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def display_workflow_guide():
    """Display workflow guide with improved styling."""
    st.markdown("<h2>Getting Started Guide</h2>", unsafe_allow_html=True)
    
    # Create tabs for workflow steps
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Data Upload", 
        "2. Feature Engineering", 
        "3. Model Training", 
        "4. Anomaly Detection"
    ])
    
    with tab1:
        st.markdown("""
        <div style='padding: 1rem;'>
            <h3>Step 1: Data Upload & Exploration</h3>
            <p>Start by uploading your network traffic data or use the sample dataset.</p>
            <ol>
                <li>Go to the <b>Data Upload & Exploration</b> page</li>
                <li>Upload a CSV file with network traffic data</li>
                <li>Explore the dataset to understand traffic patterns</li>
                <li>Select the target column containing attack labels</li>
            </ol>
            <a href="?page=data" target="_self"><button style='background-color: #1e3a5c; color: white; border: none; padding: 0.5rem 1rem; border-radius: 4px;'>Go to Data Upload</button></a>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div style='padding: 1rem;'>
            <h3>Step 2: Data Preprocessing & Feature Engineering</h3>
            <p>Prepare your data for machine learning models by cleaning and transforming features.</p>
            <ol>
                <li>Handle missing values and outliers</li>
                <li>Encode categorical features</li>
                <li>Scale numerical features</li>
                <li>Engineer advanced features to improve detection</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("""
        <div style='padding: 1rem;'>
            <h3>Step 3: Model Training & Evaluation</h3>
            <p>Train and evaluate different machine learning models to find the best performer.</p>
            <ol>
                <li>Go to the <b>Model Training & Evaluation</b> page</li>
                <li>Select models to train (Random Forest, SVM, etc.)</li>
                <li>Configure training parameters</li>
                <li>Compare model performance metrics</li>
            </ol>
            <a href="?page=training" target="_self"><button style='background-color: #1e3a5c; color: white; border: none; padding: 0.5rem 1rem; border-radius: 4px;'>Go to Model Training</button></a>
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("""
        <div style='padding: 1rem;'>
            <h3>Step 4: Anomaly Detection</h3>
            <p>Use trained models to detect anomalies in new network traffic data.</p>
            <ol>
                <li>Go to the <b>Anomaly Detection</b> page</li>
                <li>Upload new network traffic data</li>
                <li>Run the detection process</li>
                <li>Explore visualization of detected anomalies</li>
            </ol>
            <a href="?page=prediction" target="_self"><button style='background-color: #1e3a5c; color: white; border: none; padding: 0.5rem 1rem; border-radius: 4px;'>Go to Anomaly Detection</button></a>
        </div>
        """, unsafe_allow_html=True)

def show_attack_types_info():
    """Show information about different attack types with improved layout."""
    st.markdown("<h2>Detectable Attack Types</h2>", unsafe_allow_html=True)
    
    # Create a mapping of attack types to descriptions
    attack_types = {
        "DoS Attacks": {
            "description": "Denial of Service attacks attempt to make a network resource unavailable by flooding it with traffic or exploiting vulnerabilities.",
            "icon": "🚫"
        },
        "DDoS Attacks": {
            "description": "Distributed Denial of Service attacks use multiple systems to flood the target with traffic.",
            "icon": "🌐"
        },
        "Web Attacks": {
            "description": "Including SQL injection, XSS, and brute force attacks targeting web applications.",
            "icon": "🔒"
        },
        "Infiltration": {
            "description": "Malicious activities from inside the network, often after an initial penetration.",
            "icon": "🕵️"
        },
        "Port Scanning": {
            "description": "Reconnaissance technique to discover open ports and services on network hosts.",
            "icon": "🔍"
        },
        "Botnet": {
            "description": "Networks of compromised computers controlled remotely to perform coordinated attacks.",
            "icon": "🤖"
        }
    }
    
    # Display in a better grid layout
    cols = st.columns(3)
    
    for i, (attack, info) in enumerate(attack_types.items()):
        col_idx = i % 3
        with cols[col_idx]:
            st.markdown(f"""
            <div class='dashboard-card'>
                <h3 style='text-align: center;'>{info['icon']} {attack}</h3>
                <p>{info['description']}</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    # For testing the page in isolation
    import streamlit as st
    
    # Initialize session state with sample data
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'target_column' not in st.session_state:
        st.session_state.target_column = None
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None
    if 'features' not in st.session_state:
        st.session_state.features = None
    
    # Display the page
    show_home_page()