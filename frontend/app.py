import streamlit as st
import os
import sys
from pathlib import Path

# Add the project root to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import pages
from frontend.pages.home import show_home_page
from frontend.pages.data import show_data_page
from frontend.pages.training import show_training_page
from frontend.pages.prediction import show_prediction_page

# Set page config
st.set_page_config(
    page_title="Network Traffic Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_custom_css():
    """Load custom CSS from the file or use embedded CSS if file not found."""
    css_file = Path(__file__).parent / "static" / "style.css"
    
    if css_file.exists():
        with open(css_file, "r") as f:
            return f"<style>{f.read()}</style>"
    else:
        # Fallback to embedded CSS
        return """
        <style>
        /* Modern styling for Network Traffic Anomaly Detection app */
        
        /* Global styles */
        .main {
            background-color: #f9f9f9;
        }
        
        [data-testid="stSidebar"] {
            background-color: #1e3a5c;
            color: white;
        }
        
        [data-testid="stSidebar"] .sidebar-content {
            padding: 1rem;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #1e3a5c;
            font-family: 'Arial', sans-serif;
        }
        
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] .stRadio label {
            color: white !important;
        }
        
        /* Navigation */
        .stRadio label {
            color: #f9f9f9;
            font-weight: 500;
        }
        
        /* Cards */
        div.css-1r6slb0.e1tzin5v2 {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        }
        
        /* Status indicators */
        .status-indicator-success {
            color: #28a745;
            font-weight: bold;
        }
        
        .status-indicator-warning {
            color: #ffc107;
            font-weight: bold;
        }
        
        .status-indicator-danger {
            color: #dc3545;
            font-weight: bold;
        }
        
        /* Buttons */
        .stButton>button {
            background-color: #1e3a5c;
            color: white;
            border-radius: 6px;
            border: none;
            padding: 0.5rem 1rem;
        }
        
        .stButton>button:hover {
            background-color: #2c5282;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            font-size: 2rem !important;
            color: #1e3a5c;
            font-weight: bold;
        }
        
        /* Progress bar */
        .stProgress>div>div {
            background-color: #1e3a5c;
        }
        
        /* Data tables */
        .stDataFrame {
            border-radius: 5px;
            overflow: hidden;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 4px 4px 0 0;
            padding: 8px 16px;
            background-color: #f2f2f2;
        }
        
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #1e3a5c;
            color: white;
        }
        
        /* Dashboard cards */
        .dashboard-card {
            background-color: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .dashboard-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        
        /* Progress steps */
        .workflow-step {
            background-color: #e9ecef;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            border-left: 5px solid #1e3a5c;
        }
        
        .workflow-step.active {
            background-color: #e3f2fd;
            border-left: 5px solid #1e88e5;
        }
        
        .workflow-step.completed {
            background-color: #e8f5e9;
            border-left: 5px solid #43a047;
        }
        </style>
        """

st.markdown(load_custom_css(), unsafe_allow_html=True)

# Define page names and functions
PAGES = {
    "Dashboard": show_home_page,
    "Data Upload & Exploration": show_data_page,
    "Model Training & Evaluation": show_training_page,
    "Anomaly Detection": show_prediction_page
}

# Icons for each page
PAGE_ICONS = {
    "Dashboard": "📊",
    "Data Upload & Exploration": "📁",
    "Model Training & Evaluation": "🤖",
    "Anomaly Detection": "🔍"
}

# Define workflow order
WORKFLOW_ORDER = list(PAGES.keys())

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
    if 'current_page' not in st.session_state:
        st.session_state.current_page = WORKFLOW_ORDER[0]
    if 'page_history' not in st.session_state:
        st.session_state.page_history = []

def render_sidebar():
    """Render the sidebar with improved navigation and status indicators."""
    # Logo and title
    st.sidebar.markdown(f"<h1 style='text-align: center;'>🔍</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("<h2 style='text-align: center; margin-top: 0;'>Network Traffic<br>Anomaly Detection</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    
    # Navigation
    st.sidebar.markdown("### Navigation")
    
    # Determine current step in workflow
    current_step = WORKFLOW_ORDER.index(st.session_state.current_page)
    
    # Create radio buttons with current page selected
    selected_page = st.sidebar.radio(
        "Go to",
        WORKFLOW_ORDER,
        index=current_step,
        format_func=lambda x: f"{PAGE_ICONS[x]} {x}"
    )
    
    # Update current page if changed
    if selected_page != st.session_state.current_page:
        st.session_state.page_history.append(st.session_state.current_page)
        if len(st.session_state.page_history) > 5:
            st.session_state.page_history.pop(0)
        st.session_state.current_page = selected_page
        st.experimental_rerun()
    
    # Display workflow progress
    st.sidebar.markdown("### Workflow Progress")
    
    # Determine status for each step
    data_status = "completed" if st.session_state.data is not None else "pending"
    model_status = "completed" if st.session_state.trained_models else "pending"
    detection_status = "completed" if ('prediction_results' in st.session_state and 
                                       st.session_state.prediction_results is not None) else "pending"
    
    # Set current step as active
    workflow_statuses = {
        "Dashboard": "info",
        "Data Upload & Exploration": data_status,
        "Model Training & Evaluation": model_status,
        "Anomaly Detection": detection_status
    }
    
    # Current page is always active
    workflow_statuses[st.session_state.current_page] = "active"
    
    # Display workflow steps with status indicators
    for page in WORKFLOW_ORDER:
        status = workflow_statuses[page]
        icon = "✅" if status == "completed" else "🔄" if status == "active" else "⏳"
        st.sidebar.markdown(
            f"<div style='padding: 5px; opacity: {'1' if status in ['active', 'completed'] else '0.7'};'>"
            f"{icon} {page}</div>", 
            unsafe_allow_html=True
        )
    
    # Status section
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown("### System Status")
    
    # Data status
    data_indicator = "status-indicator-success" if st.session_state.data is not None else "status-indicator-danger"
    data_icon = "✅" if st.session_state.data is not None else "❌"
    data_message = "Data loaded" if st.session_state.data is not None else "No data loaded"
    st.sidebar.markdown(f"<span class='{data_indicator}'>{data_icon} {data_message}</span>", unsafe_allow_html=True)
    
    # Model status
    model_indicator = "status-indicator-success" if st.session_state.trained_models else "status-indicator-danger"
    model_icon = "✅" if st.session_state.trained_models else "❌"
    model_message = "Model trained" if st.session_state.trained_models else "No model trained"
    st.sidebar.markdown(f"<span class='{model_indicator}'>{model_icon} {model_message}</span>", unsafe_allow_html=True)
    
    # Prediction status
    pred_status = 'prediction_results' in st.session_state and st.session_state.prediction_results is not None
    pred_indicator = "status-indicator-success" if pred_status else "status-indicator-danger"
    pred_icon = "✅" if pred_status else "❌"
    pred_message = "Predictions made" if pred_status else "No predictions"
    st.sidebar.markdown(f"<span class='{pred_indicator}'>{pred_icon} {pred_message}</span>", unsafe_allow_html=True)
    
    # Quick action buttons
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown("### Quick Actions")
    
    # Columns for action buttons
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("📁 Upload Data"):
            st.session_state.current_page = "Data Upload & Exploration"
            st.experimental_rerun()
    
    with col2:
        if st.button("🔍 Detect"):
            st.session_state.current_page = "Anomaly Detection" 
            st.experimental_rerun()
    
    # Footer
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown(
        "<div style='text-align: center; opacity: 0.7; font-size: 0.8em;'>"
        "Network Anomaly Detection System<br>"
        "v1.0.0"
        "</div>",
        unsafe_allow_html=True
    )

def main():
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Display the selected page
    page_function = PAGES[st.session_state.current_page]
    page_function()

if __name__ == "__main__":
    main()