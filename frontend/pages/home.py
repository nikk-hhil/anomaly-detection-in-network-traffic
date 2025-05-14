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
    # Page title and introduction
    st.title("🔍 Network Traffic Anomaly Detection")
    
    # Introduction section
    st.markdown("""
    ## Welcome to the Network Traffic Anomaly Detection System
    
    This application uses machine learning to identify and classify network traffic anomalies 
    and potential cyber attacks by analyzing network flow features.
    
    ### Key Features
    - **Data Analysis**: Upload and explore network traffic data
    - **Advanced Feature Engineering**: Create 60+ engineered features from raw network data
    - **Multiple ML Models**: Train and evaluate various classification algorithms
    - **Real-time Detection**: Identify anomalies in network traffic
    """)
    
    # Create columns for dashboard cards
    col1, col2 = st.columns(2)
    
    with col1:
        show_data_status_card()
    
    with col2:
        show_model_status_card()
    
    # Show attack types information
    st.markdown("---")
    show_attack_types_info()
    
    # Show getting started guide
    st.markdown("---")
    show_getting_started_guide()

def show_data_status_card():
    """Show data status information card."""
    st.markdown("### 📊 Data Status")
    
    if st.session_state.data is not None:
        # Display info about loaded data
        data = st.session_state.data
        st.success(f"✅ Data loaded: {data.shape[0]} records with {data.shape[1]} features")
        
        # Show a small sample of the dataset classes if target is identified
        if st.session_state.target_column:
            target_col = st.session_state.target_column
            class_counts = data[target_col].value_counts()
            
            # Create a simple Plotly pie chart of the classes
            fig = px.pie(
                values=class_counts.values,
                names=class_counts.index,
                title="Distribution of Traffic Classes",
                hole=0.4,
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No data loaded yet")
        st.markdown("""
        Start by going to the **Data Upload & Exploration** page to:
        - Upload network traffic data
        - Explore features and distributions
        - Preprocess the data for model training
        """)

def show_model_status_card():
    """Show model status information card."""
    st.markdown("### 🤖 Model Status")
    
    if st.session_state.trained_models:
        # Display info about trained models
        num_models = len(st.session_state.trained_models)
        st.success(f"✅ {num_models} models trained")
        
        if st.session_state.evaluation_results:
            # Create a comparison chart of model performance
            models = list(st.session_state.trained_models.keys())
            
            # Create sample metrics for visualization (replace with actual metrics if available)
            if st.session_state.evaluation_results:
                metrics = st.session_state.evaluation_results
                
                # Create a radar chart for model comparison
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
                            name=model
                        ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No models trained yet")
        st.markdown("""
        After loading data, go to the **Model Training & Evaluation** page to:
        - Train different types of models
        - Compare model performance
        - Select the best model for anomaly detection
        """)

def show_attack_types_info():
    """Show information about different attack types."""
    st.markdown("### 🛡️ Detectable Attack Types")
    
    # Create a mapping of attack types to descriptions
    attack_types = {
        "DoS Attacks": "Denial of Service attacks attempt to make a network resource unavailable by flooding it with traffic or exploiting vulnerabilities.",
        "DDoS Attacks": "Distributed Denial of Service attacks use multiple systems to flood the target with traffic.",
        "Web Attacks": "Including SQL injection, XSS, and brute force attacks targeting web applications.",
        "Infiltration": "Malicious activities from inside the network, often after an initial penetration.",
        "Port Scanning": "Reconnaissance technique to discover open ports and services on network hosts.",
        "Botnet": "Networks of compromised computers controlled remotely to perform coordinated attacks."
    }
    
    # Display in a multi-column layout
    cols = st.columns(3)
    
    for i, (attack, description) in enumerate(attack_types.items()):
        col_idx = i % 3
        with cols[col_idx]:
            st.markdown(f"**{attack}**")
            st.markdown(f"{description}")
            st.markdown("---")

def show_getting_started_guide():
    """Show getting started guide."""
    st.markdown("### 🚀 Getting Started")
    
    # Create tabs for different steps
    tab1, tab2, tab3, tab4 = st.tabs(["1. Data Upload", "2. Data Exploration", "3. Model Training", "4. Anomaly Detection"])
    
    with tab1:
        st.markdown("""
        - Go to the **Data Upload & Exploration** page
        - Upload your network traffic CSV file
        - The system will automatically analyze the dataset
        - Identify the target column containing attack labels
        """)
    
    with tab2:
        st.markdown("""
        - Explore feature correlations and distributions
        - View class distribution of attack types
        - Preprocess data to handle missing values and encode categorical features
        - Generate advanced features to improve detection
        """)
    
    with tab3:
        st.markdown("""
        - Go to the **Model Training & Evaluation** page
        - Select models to train (Random Forest, SVM, etc.)
        - Configure model hyperparameters
        - Compare model performance metrics
        - Select the best model for detection
        """)
    
    with tab4:
        st.markdown("""
        - Go to the **Anomaly Detection** page
        - Upload new network traffic data for analysis
        - View detected anomalies and attack types
        - Explore visualization of anomalous traffic patterns
        - Export detection results
        """)

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
    
    # Display the page
    show_home_page()