"""
Configuration settings for the Network Traffic Anomaly Detection frontend.
"""

import os

# Application settings
APP_TITLE = "Network Traffic Anomaly Detection"
APP_ICON = "🔍"
APP_LAYOUT = "wide"
APP_INITIAL_SIDEBAR_STATE = "expanded"

# Path settings
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
TEMP_DIR = os.path.join(ROOT_DIR, "temp")

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, TEMP_DIR]:
    os.makedirs(directory, exist_ok=True)

# File paths
CSS_PATH = os.path.join(STATIC_DIR, "style.css")
SAMPLE_DATA_PATH = os.path.join(DATA_DIR, "sample_network_traffic.csv")
SAMPLE_TEST_PATH = os.path.join(DATA_DIR, "sample_test_traffic.csv")

# Model settings
DEFAULT_MODELS = ["logistic_regression", "decision_tree"]
AVAILABLE_MODELS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "svm",
    "knn",
    "mlp",
    "adaboost"
]

# UI settings
THEME_COLOR = "#3498db"
METRICS_PRECISION = 4  # Number of decimal places for metrics

# Load custom CSS
def load_custom_css():
    """Load custom CSS for the application."""
    if os.path.exists(CSS_PATH):
        with open(CSS_PATH, "r") as f:
            return f"<style>{f.read()}</style>"
    return ""