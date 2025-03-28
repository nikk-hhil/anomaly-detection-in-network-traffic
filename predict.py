import argparse
import os
import logging
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.anomaly_detector import AnomalyDetector
from src.utils import save_results, load_results, plot_threshold_metrics
from src.visualizer import DataVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("network_anomaly_prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Network Traffic Anomaly Detector')
    
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input CSV file')
    parser.add_argument('--output', type=str, default='./results',
                        help='Directory to save prediction results')
    parser.add_argument('--model', type=str, default='./models/best_model.joblib',
                        help='Path to trained model file')
    parser.add_argument('--preprocessor', type=str, default='./models/preprocessor.joblib',
                        help='Path to preprocessor file')
    parser.add_argument('--feature-engineer', type=str, default='./models/feature_engineer.joblib',
                        help='Path to feature engineer file')
    parser.add_argument('--target-column', type=str, default=None,
                        help='Name of target column (if present for evaluation)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for anomaly detection')
    parser.add_argument('--batch-size', type=int, default=10000,
                        help='Batch size for processing large files')
    parser.add_argument('--explain', action='store_true',
                        help='Generate explanations for predictions')
    parser.add_argument('--explain-samples', type=int, default=10,
                        help='Number of random samples to explain')
    parser.add_argument('--optimize-threshold', action='store_true',
                        help='Optimize the threshold based on evaluation metrics')
    
    return parser.parse_args()

def optimize_detection_threshold(detector: AnomalyDetector, data: pd.DataFrame, 
                                target_column: str) -> float:
    """
    Find optimal detection threshold based on F1 score.
    
    Parameters:
    -----------
    detector : AnomalyDetector
        Anomaly detector instance
    data : pd.DataFrame
        Evaluation data
    target_column : str
        Name of target column
        
    Returns:
    --------
    float
        Optimal threshold
    """
    from sklearn.metrics import f1_score
    
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    # Get true labels
    y_true = data[target_column].values
    
    # Encode labels if necessary
    if detector.inverse_label_mapping and y_true.dtype == object:
        y_true_encoded = np.array([detector.inverse_label_mapping.get(label, -1) for label in y_true])
        if -1 in y_true_encoded:
            logger.warning("Some labels in test data were not seen during training")
        y_true = y_true_encoded
    
    # Preprocess data
    X_preprocessed = detector.preprocess_data(data, target_column)
    
    # Get feature names
    feature_names = [col for col in data.columns if col != target_column]
    
    # Engineer features
    if detector.feature_engineer is not None:
        try:
            X_final = detector.engineer_features(X_preprocessed, feature_names)
        except Exception as e:
            logger.warning(f"Feature engineering failed: {e}. Using preprocessed features.")
            X_final = X_preprocessed
    else:
        X_final = X_preprocessed
    
    # Get probabilities
    y_proba = detector.model.predict_proba(X_final)
    
    if y_proba.shape[1] == 2:  # Binary classification
        y_proba_positive = y_proba[:, 1]
    else:  # Multiclass classification
        # For multiclass, optimize for each class vs rest
        logger.info("Multiclass optimization not implemented. Using default threshold.")
        return detector.threshold
    
    # Try different thresholds
    thresholds = np.arange(0.1, 0.95, 0.05)
    best_threshold = detector.threshold
    best_f1 = 0
    
    for threshold in thresholds:
        # Apply threshold
        y_pred = (y_proba_positive >= threshold).astype(int)
        
        # Calculate F1 score
        try:
            f1 = f1_score(y_true, y_pred)
            logger.info(f"Threshold: {threshold:.2f}, F1 score: {f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        except Exception as e:
            logger.warning(f"Error calculating F1 score for threshold {threshold}: {e}")
    
    logger.info(f"Optimal threshold: {best_threshold:.2f} (F1 score: {best_f1:.4f})")
    
    # Optional: Visualize threshold metrics
    plot_threshold_metrics(y_true, y_proba_positive, 
                          save_path=os.path.join(detector.output_dir, "threshold_metrics.png"))
    
    return best_threshold

def generate_explanations(detector: AnomalyDetector, data: pd.DataFrame, 
                         target_column: Optional[str], n_samples: int = 10,
                         output_dir: str = './results') -> List[Dict[str, Any]]:
    """
    Generate explanations for a subset of predictions.
    
    Parameters:
    -----------
    detector : AnomalyDetector
        Anomaly detector instance
    data : pd.DataFrame
        Input data
    target_column : Optional[str]
        Name of target column (if present)
    n_samples : int
        Number of random samples to explain
    output_dir : str
        Directory to save explanations
        
    Returns:
    --------
    List[Dict[str, Any]]
        List of explanations
    """
    # Get predictions first
    y_pred, _ = detector.predict(data, target_column)
    
    # For anomaly detection, focus on explaining anomalies
    anomaly_indices = np.where(y_pred == 1)[0]
    
    if len(anomaly_indices) == 0:
        logger.warning("No anomalies detected. Explaining random samples instead.")
        sample_indices = np.random.choice(len(data), min(n_samples, len(data)), replace=False)
    else:
        # Sample from anomalies (or all if fewer than n_samples)
        if len(anomaly_indices) > n_samples:
            sample_indices = np.random.choice(anomaly_indices, n_samples, replace=False)
        else:
            sample_indices = anomaly_indices
    
    # Generate explanations
    explanations = []
    for idx in sample_indices:
        try:
            explanation = detector.explain_prediction(data, idx, target_column)
            explanations.append(explanation)
        except Exception as e:
            logger.error(f"Error explaining prediction for row {idx}: {e}")
    
    # Save explanations
    if explanations:
        os.makedirs(output_dir, exist_ok=True)
        explanations_path = os.path.join(output_dir, "explanations.json")
        save_results({"explanations": explanations}, explanations_path)
        logger.info(f"Saved {len(explanations)} explanations to {explanations_path}")
    
    return explanations

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # Initialize anomaly detector
        detector = AnomalyDetector(
            model_path=args.model,
            preprocessor_path=args.preprocessor,
            feature_engineer_path=args.feature_engineer,
            threshold=args.threshold
        )
        
        # Check if input file exists
        if not os.path.exists(args.input):
            logger.error(f"Input file not found: {args.input}")
            return
        
        # Log processing information
        logger.info(f"Processing file: {args.input}")
        logger.info(f"Using model: {args.model}")
        logger.info(f"Detection threshold: {args.threshold}")
        
        # Determine output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = os.path.splitext(os.path.basename(args.input))[0]
        output_path = os.path.join(args.output, f"{output_base}_predictions_{timestamp}.csv")
        
        # Check if target column is available for evaluation
        if args.target_column:
            logger.info(f"Target column provided: {args.target_column}")
            
            # Try loading a small sample to check if target column exists
            sample_data = pd.read_csv(args.input, nrows=10)
            if args.target_column not in sample_data.columns:
                logger.warning(f"Target column '{args.target_column}' not found in data. Skipping evaluation.")
                args.target_column = None
            
            # Optimize threshold if requested and target column is available
            if args.optimize_threshold and args.target_column:
                logger.info("Optimizing detection threshold...")
                
                # Load evaluation data (limit size for optimization)
                eval_data = pd.read_csv(args.input, nrows=min(10000, args.batch_size*2))
                
                # Find optimal threshold
                optimal_threshold = optimize_detection_threshold(detector, eval_data, args.target_column)
                
                # Update detector threshold
                detector.set_threshold(optimal_threshold)
                
                logger.info(f"Using optimized threshold: {optimal_threshold:.4f}")
        
        # Process data and make predictions
        logger.info("Processing data and making predictions...")
        
        # For large files, use batch processing
        file_size = os.path.getsize(args.input) / (1024 * 1024)  # Size in MB
        
        if file_size > 100:  # If file is larger than 100MB
            logger.info(f"Large file detected ({file_size:.2f} MB). Processing in batches.")
            detector.predict_batch(args.input, output_path, args.target_column, args.batch_size)
        else:
            # Load entire file
            data = pd.read_csv(args.input)
            logger.info(f"Loaded data with shape: {data.shape}")
            
            # Make predictions
            y_pred, y_proba = detector.predict(data, args.target_column)
            
            # Create result dataframe
            result = data.copy()
            result['predicted_class'] = y_pred
            
            # Add class names if available
            if detector.label_mapping:
                result['predicted_label'] = [detector._get_class_name(pred) for pred in y_pred]
            
            # Add probabilities if available
            if y_proba is not None:
                if y_proba.ndim > 1:
                    # Multi-class case
                    for j in range(y_proba.shape[1]):
                        result[f'probability_class_{j}'] = y_proba[:, j]
                        if detector.label_mapping:
                            class_name = detector._get_class_name(j)
                            result[f'probability_{class_name}'] = y_proba[:, j]
                else:
                    # Binary case
                    result['probability'] = y_proba
            
            # Save results
            result.to_csv(output_path, index=False)
            logger.info(f"Saved predictions to {output_path}")
            
            # Evaluate if target column is available
            if args.target_column:
                logger.info("Evaluating predictions...")
                metrics = detector.evaluate(data, args.target_column)
                
                # Save metrics
                metrics_path = os.path.join(args.output, f"{output_base}_metrics_{timestamp}.json")
                save_results(metrics, metrics_path)
                logger.info(f"Saved evaluation metrics to {metrics_path}")
                
                # Create visualizations
                visualizer = DataVisualizer(output_dir=args.output)
                
                # Confusion matrix
                if 'confusion_matrix' in metrics:
                    class_names = [detector._get_class_name(i) for i in range(len(metrics['confusion_matrix']))]
                    try:
                        # You need to implement this in your visualizer
                        visualizer.plot_confusion_matrix(
                            np.array(metrics['confusion_matrix']),
                            class_names=class_names,
                            title="Confusion Matrix",
                            save_fig=True
                        )
                    except Exception as e:
                        logger.warning(f"Error plotting confusion matrix: {e}")
        
        # Generate explanations if requested
        if args.explain:
            logger.info(f"Generating explanations for {args.explain_samples} samples...")
            
            # Load data if not already loaded
            if 'data' not in locals():
                # Load a subset for explanation
                data = pd.read_csv(args.input, nrows=min(args.batch_size * 2, 20000))
            
            # Generate explanations
            explanations = generate_explanations(
                detector, data, args.target_column, 
                n_samples=args.explain_samples, 
                output_dir=args.output
            )
            
            logger.info(f"Generated {len(explanations)} explanations")
        
        logger.info("Processing completed successfully")
    
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()