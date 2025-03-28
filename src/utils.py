import numpy as np
import pandas as pd
import os
import time
import datetime
import json
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

logger = logging.getLogger(__name__)

def timer_decorator(func):
    """
    Decorator to time function execution.
    
    Parameters:
    -----------
    func : callable
        Function to time
    
    Returns:
    --------
    callable
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Function '{func.__name__}' executed in {execution_time:.2f} seconds")
        return result
    return wrapper

def save_results(results: Dict, output_path: str) -> None:
    """
    Save evaluation results to JSON.
    
    Parameters:
    -----------
    results : Dict
        Results to save
    output_path : str
        Path to save results
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Serialize numpy arrays and other non-serializable objects
    def serialize_item(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        elif isinstance(item, (np.int64, np.int32, np.float64, np.float32)):
            return item.item()
        elif isinstance(item, dict):
            return {k: serialize_item(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [serialize_item(i) for i in item]
        else:
            return item
    
    serialized_results = serialize_item(results)
    
    # Add timestamp
    serialized_results['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(serialized_results, f, indent=4)
    
    logger.info(f"Results saved to {output_path}")

def load_results(input_path: str) -> Dict:
    """
    Load results from JSON.
    
    Parameters:
    -----------
    input_path : str
        Path to load results from
        
    Returns:
    --------
    Dict
        Loaded results
    """
    with open(input_path, 'r') as f:
        results = json.load(f)
    
    logger.info(f"Results loaded from {input_path}")
    return results

def find_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray, 
                          metric: str = 'f1') -> Tuple[float, Dict[str, float]]:
    """
    Find optimal decision threshold for binary classification.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels
    y_score : np.ndarray
        Target scores (probabilities)
    metric : str
        Metric to optimize ('f1', 'precision', 'recall', 'accuracy', 'balanced_accuracy')
        
    Returns:
    --------
    Tuple[float, Dict[str, float]]
        Optimal threshold and performance metrics at that threshold
    """
    from sklearn.metrics import (
        f1_score, precision_score, recall_score, 
        accuracy_score, balanced_accuracy_score
    )
    
    # Ensure binary labels
    if len(np.unique(y_true)) != 2:
        raise ValueError("This function only works for binary classification")
    
    # Get ROC curve points
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    # Initialize variables
    best_threshold = 0.5  # Default threshold
    best_score = 0.0
    
    # Calculate metric for each threshold
    metrics = {}
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred)
        elif metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        elif metric == 'balanced_accuracy':
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Update best threshold if score is better
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    # Calculate metrics at best threshold
    y_pred_best = (y_score >= best_threshold).astype(int)
    metrics = {
        'threshold': best_threshold,
        'f1': f1_score(y_true, y_pred_best),
        'precision': precision_score(y_true, y_pred_best),
        'recall': recall_score(y_true, y_pred_best),
        'accuracy': accuracy_score(y_true, y_pred_best),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred_best)
    }
    
    return best_threshold, metrics

def plot_threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, save_path: Optional[str] = None) -> None:
    """
    Plot metrics vs threshold for binary classification.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels
    y_score : np.ndarray
        Target scores (probabilities)
    save_path : Optional[str]
        Path to save the plot
    """
    from sklearn.metrics import (
        f1_score, precision_score, recall_score, 
        accuracy_score, balanced_accuracy_score
    )
    
    # Ensure binary labels
    if len(np.unique(y_true)) != 2:
        raise ValueError("This function only works for binary classification")
    
    # Thresholds to evaluate
    thresholds = np.linspace(0.01, 0.99, 99)
    
    # Calculate metrics for each threshold
    f1_scores = []
    precision_scores = []
    recall_scores = []
    accuracy_scores = []
    balanced_accuracy_scores = []
    
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        
        f1_scores.append(f1_score(y_true, y_pred))
        precision_scores.append(precision_score(y_true, y_pred))
        recall_scores.append(recall_score(y_true, y_pred))
        accuracy_scores.append(accuracy_score(y_true, y_pred))
        balanced_accuracy_scores.append(balanced_accuracy_score(y_true, y_pred))
    
    # Find optimal thresholds for each metric
    best_threshold_f1 = thresholds[np.argmax(f1_scores)]
    best_threshold_balanced_acc = thresholds[np.argmax(balanced_accuracy_scores)]
    
    # Plot metrics vs threshold
    plt.figure(figsize=(12, 8))
    
    plt.plot(thresholds, f1_scores, label='F1 Score')
    plt.plot(thresholds, precision_scores, label='Precision')
    plt.plot(thresholds, recall_scores, label='Recall')
    plt.plot(thresholds, accuracy_scores, label='Accuracy')
    plt.plot(thresholds, balanced_accuracy_scores, label='Balanced Accuracy')
    
    # Add vertical lines at optimal thresholds
    plt.axvline(x=best_threshold_f1, color='r', linestyle='--', 
               alpha=0.5, label=f'Best F1 @ {best_threshold_f1:.2f}')
    plt.axvline(x=best_threshold_balanced_acc, color='g', linestyle='--', 
               alpha=0.5, label=f'Best Balanced Acc @ {best_threshold_balanced_acc:.2f}')
    
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def check_dataset_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame, 
                       categorical_cols: List[str] = None, 
                       threshold: float = 0.1) -> Dict[str, float]:
    """
    Check for dataset drift between reference and current data.
    
    Parameters:
    -----------
    reference_data : pd.DataFrame
        Reference dataset (e.g., training data)
    current_data : pd.DataFrame
        Current dataset to check for drift
    categorical_cols : List[str]
        List of categorical column names
    threshold : float
        Threshold for flagging significant drift
        
    Returns:
    --------
    Dict[str, float]
        Dictionary with drift metrics for each column
    """
    if categorical_cols is None:
        categorical_cols = []
    
    # Initialize results dictionary
    drift_results = {}
    significant_drift_cols = []
    
    # Check common columns
    common_cols = set(reference_data.columns).intersection(set(current_data.columns))
    
    for col in common_cols:
        # Skip if too many missing values
        if (reference_data[col].isnull().mean() > 0.5 or 
            current_data[col].isnull().mean() > 0.5):
            drift_results[col] = {'drift_score': None, 'status': 'Too many missing values'}
            continue
        
        # For categorical columns
        if col in categorical_cols or reference_data[col].dtype == 'object':
            # Compare distribution using chi-squared test
            from scipy.stats import chi2_contingency
            
            # Get value counts
            ref_counts = reference_data[col].value_counts().to_dict()
            curr_counts = current_data[col].value_counts().to_dict()
            
            # Get all unique values
            all_values = set(ref_counts.keys()).union(set(curr_counts.keys()))
            
            # Build contingency table
            table = []
            for val in all_values:
                ref_val_count = ref_counts.get(val, 0)
                curr_val_count = curr_counts.get(val, 0)
                table.append([ref_val_count, curr_val_count])
            
            # Perform chi-squared test
            try:
                # Catch empty table error
                if len(table) == 0 or sum(sum(row) for row in table) == 0:
                    drift_results[col] = {'drift_score': None, 'status': 'Empty contingency table'}
                    continue
                
                chi2, p_value, _, _ = chi2_contingency(table)
                
                # Normalize by number of samples to get comparable metric
                drift_score = chi2 / (len(reference_data) + len(current_data))
                
                drift_results[col] = {
                    'drift_score': drift_score,
                    'p_value': p_value,
                    'status': 'Significant drift' if p_value < 0.05 else 'No significant drift'
                }
                
                if p_value < 0.05:
                    significant_drift_cols.append(col)
                
            except Exception as e:
                drift_results[col] = {'drift_score': None, 'status': f'Error: {str(e)}'}
        
        # For numerical columns
        else:
            try:
                # Use Kolmogorov-Smirnov test
                from scipy.stats import ks_2samp
                
                # Skip if not enough data
                if len(reference_data[col].dropna()) < 10 or len(current_data[col].dropna()) < 10:
                    drift_results[col] = {'drift_score': None, 'status': 'Not enough data'}
                    continue
                
                # Perform KS test
                ks_stat, p_value = ks_2samp(
                    reference_data[col].dropna(), 
                    current_data[col].dropna()
                )
                
                drift_results[col] = {
                    'drift_score': ks_stat,
                    'p_value': p_value,
                    'status': 'Significant drift' if p_value < 0.05 else 'No significant drift'
                }
                
                if p_value < 0.05:
                    significant_drift_cols.append(col)
                
            except Exception as e:
                drift_results[col] = {'drift_score': None, 'status': f'Error: {str(e)}'}
    
    # Summary
    drift_results['summary'] = {
        'total_columns_checked': len(common_cols),
        'columns_with_significant_drift': len(significant_drift_cols),
        'drift_ratio': len(significant_drift_cols) / len(common_cols) if len(common_cols) > 0 else 0,
        'drifted_columns': significant_drift_cols
    }
    
    return drift_results

def get_model_size(model: Any) -> str:
    """
    Get the memory size of a model.
    
    Parameters:
    -----------
    model : Any
        Model object
        
    Returns:
    --------
    str
        Size of the model in human-readable format
    """
    import sys
    import pickle
    
    # Serialize the model
    serialized_model = pickle.dumps(model)
    
    # Get size in bytes
    size_bytes = sys.getsizeof(serialized_model)
    
    # Convert to human-readable format
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"