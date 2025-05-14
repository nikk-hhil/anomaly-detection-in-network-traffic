import streamlit as st
import os
import sys

# Add the project root to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
from frontend.config import APP_TITLE, APP_ICON, APP_LAYOUT, APP_INITIAL_SIDEBAR_STATE, load_custom_css

# Import pages
from frontend.pages.home import show_home_page
from frontend.pages.data import show_data_page
from frontend.pages.training import show_training_page
from frontend.pages.prediction import show_prediction_page

# Set page config
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=APP_LAYOUT,
    initial_sidebar_state=APP_INITIAL_SIDEBAR_STATE
)

# Load custom CSS
st.markdown(load_custom_css(), unsafe_allow_html=True)

# Define page names and functions
PAGES = {
    "Home": show_home_page,
    "Data Upload & Exploration": show_data_page,
    "Model Training & Evaluation": show_training_page,
    "Anomaly Detection": show_prediction_page
}

def initialize_session_state():
    """Initialize session state variables."""
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'preprocessed_data' not in st.session_state:
        st.session_state.preprocessed_data = None
    if 'features' not in st.session_state:
        st.session_state.features = None
    if 'target_column' not in st.session_state:
        st.session_state.target_column = None
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'best_model' not in st.session_state:
        st.session_state.best_model = None
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'page_history' not in st.session_state:
        st.session_state.page_history = []

def main():
    # Initialize session state
    initialize_session_state()
    
    # Create sidebar for navigation
    st.sidebar.title("Navigation")
    
    # Add logo or image to sidebar
    st.sidebar.image("https://img.icons8.com/clouds/100/000000/network-drive.png", width=100)
    
    # Navigation selection
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    
    # Track page navigation for breadcrumbs
    if 'current_page' not in st.session_state or st.session_state.current_page != selection:
        if 'current_page' in st.session_state:
            st.session_state.page_history.append(st.session_state.current_page)
            # Keep only the last 5 pages in history
            if len(st.session_state.page_history) > 5:
                st.session_state.page_history.pop(0)
        st.session_state.current_page = selection
    
    # Add sidebar info
    st.sidebar.info(
        "This application demonstrates network traffic anomaly detection using machine learning."
    )
    
    # Add status indicators
    st.sidebar.markdown("### Status")
    
    # Status for data
    data_status = "✅ Data loaded" if st.session_state.data is not None else "❌ No data loaded"
    # Status for model
    model_status = "✅ Model trained" if 'trained_models' in st.session_state and st.session_state.trained_models else "❌ No model trained"
    # Status for predictions
    prediction_status = "✅ Predictions made" if 'prediction_results' in st.session_state and st.session_state.prediction_results is not None else "❌ No predictions"
    
    st.sidebar.markdown(data_status)
    st.sidebar.markdown(model_status)
    st.sidebar.markdown(prediction_status)
    
    # Add page history (breadcrumbs)
    if st.session_state.page_history:
        st.sidebar.markdown("### Recent Pages")
        for prev_page in reversed(st.session_state.page_history):
            st.sidebar.markdown(f"- {prev_page}")
    
    # Add sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Created with Streamlit and Plotly"
    )
    
    # Display the selected page
    page = PAGES[selection]
    page()

if __name__ == "__main__":
    main()