import os
import sys
import pandas as pd
import numpy as np
import joblib
import logging
import argparse
from datetime import datetime
from src.anomaly_detector import AnomalyDetector 
# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the new anomaly detector


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simple_prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Simple Network Traffic Anomaly Detector')
    
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
    parser.add_argument('--target-column', type=str, default=' Label',
                        help='Name of target column (if present for evaluation)')
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # Check if input file exists
        if not os.path.exists(args.input):
            logger.error(f"Input file not found: {args.input}")
            return
        
        # Check if model files exist
        if not os.path.exists(args.model):
            logger.error(f"Model file not found: {args.model}")
            return
        if not os.path.exists(args.preprocessor):
            logger.error(f"Preprocessor file not found: {args.preprocessor}")
            return
        if not os.path.exists(args.feature_engineer):
            logger.error(f"Feature engineer file not found: {args.feature_engineer}")
            return
        
        # Initialize anomaly detector
        logger.info(f"Initializing anomaly detector...")
        detector = AnomalyDetector(
            model_path=args.model,
            preprocessor_path=args.preprocessor,
            feature_engineer_path=args.feature_engineer,
            threshold=0.5,
            output_dir=args.output
        )
        
        # Load data
        logger.info(f"Loading data from {args.input}...")
        data = pd.read_csv(args.input)
        logger.info(f"Loaded data with shape: {data.shape}")
        
        # Check if target column exists
        has_target = args.target_column in data.columns
        if has_target:
            logger.info(f"Target column '{args.target_column}' found in data")
        else:
            logger.info(f"Target column '{args.target_column}' not found in data. Will only make predictions.")
            args.target_column = None
        
        # Make predictions
        logger.info("Making predictions...")
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = os.path.splitext(os.path.basename(args.input))[0]
        output_path = os.path.join(args.output, f"{output_base}_predictions_{timestamp}.csv")
        
        result.to_csv(output_path, index=False)
        logger.info(f"Saved predictions to {output_path}")
        
        # Evaluate if target column is available
        if has_target:
            logger.info("Evaluating predictions...")
            metrics = detector.evaluate(data, args.target_column)
            
            # Save metrics
            metrics_path = os.path.join(args.output, f"{output_base}_metrics_{timestamp}.json")
            import json
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Saved evaluation metrics to {metrics_path}")
        
        logger.info("Processing completed successfully")
    
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()