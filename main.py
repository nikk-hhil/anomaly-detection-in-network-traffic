import os
import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, List, Tuple
import argparse
# Import project modules
from src.data_loader import load_dataset, get_dataset_info, save_dataset, get_column_statistics
from src.preprocessor import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from src.visualizer import DataVisualizer

# New base directory
PROJECT_BASE_DIR = "C:\\Users\\khatr\\OneDrive\\Documents\\InternshipProjects\\Anomaly detection\\anomaly-detection-in-network-traffic"

# Configure logging with absolute path
log_path = os.path.join(PROJECT_BASE_DIR, "network_anomaly_detection.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Network Traffic Anomaly Detection Pipeline')
    
    # Update default paths to use absolute paths with the new location
    parser.add_argument('--data-dir', type=str, 
                        default=os.path.join(PROJECT_BASE_DIR, 'data'), 
                        help='Directory containing the dataset files')
    parser.add_argument('--output-dir', type=str, 
                        default=os.path.join(PROJECT_BASE_DIR, 'models'),
                        help='Directory to save models and results')
    parser.add_argument('--target-column', type=str, default=None,
                        help='Name of the target column in the dataset')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of the dataset to include in the test split')
    parser.add_argument('--num-features', type=int, default=20,
                        help='Number of features to select')
    parser.add_argument('--models', type=str, default='random_forest,gradient_boosting,logistic_regression',
                        help='Comma-separated list of models to train')
    parser.add_argument('--skip-preprocessing', action='store_true',
                        help='Skip preprocessing if already done')
    parser.add_argument('--skip-feature-engineering', action='store_true',
                        help='Skip feature engineering if already done')
    
    return parser.parse_args()


def main():
    """Main execution function for the anomaly detection pipeline."""
    
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # 1. Set up project directories with absolute paths
        base_dir = PROJECT_BASE_DIR
        data_dir = args.data_dir
        models_dir = args.output_dir
        processed_data_path = os.path.join(data_dir, "processed", "merged_data.csv")
        
        # Create directories if they don't exist
        os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        
        # Rest of the code remains the same
        # 2. Load and merge dataset
        logger.info("=== Step 1: Loading dataset ===")
        try:
            # Check if merged file already exists
            if os.path.exists(processed_data_path):
                logger.info(f"Loading existing merged dataset from {processed_data_path}")
                data = pd.read_csv(processed_data_path)
                logger.info(f"Loaded dataset shape: {data.shape}")
            else:
                logger.info("Merged dataset not found. Loading and merging raw datasets...")
                # Load data from the specified directory
                data = load_dataset(data_dir)
                # Save the merged dataset
                save_dataset(data, processed_data_path)
            
            # Get dataset info and identify target column
            target_column = args.target_column or get_dataset_info(data)
            
            if target_column is None:
                logger.warning("No target column automatically identified.")
                target_column = input("Enter target column name: ")
                
                if target_column not in data.columns:
                    raise ValueError(f"Column '{target_column}' not found in the dataset")
            
            logger.info(f"\nUsing '{target_column}' as the target column")
            
            # 3. Preprocess the data
            logger.info("\n=== Step 2: Preprocessing data ===")
            
            # Save intermediate files for reproducibility
            preprocessed_data_path = os.path.join(data_dir, "processed", "preprocessed_data.npz")
            
            if os.path.exists(preprocessed_data_path) and args.skip_preprocessing:
                logger.info(f"Loading preprocessed data from {preprocessed_data_path}")
                loaded_data = np.load(preprocessed_data_path)
                X_scaled = loaded_data['X']
                y_encoded = loaded_data['y']
                feature_names = loaded_data['feature_names']
                preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
            else:
                preprocessor = DataPreprocessor()
                
                # Clean the data
                cleaned_data = preprocessor.clean_data(data, target_column)
                
                # Encode categorical features
                encoded_data = preprocessor.encode_categorical(cleaned_data, target_column)
                
                # Scale features and get encoded target
                X_scaled, y_encoded = preprocessor.scale_features(encoded_data, target_column)
                
                # Get feature names (without target column)
                feature_names = np.array(encoded_data.drop(columns=[target_column]).columns)
                
                # Save preprocessed data
                np.savez(
                    preprocessed_data_path, 
                    X=X_scaled, 
                    y=y_encoded, 
                    feature_names=feature_names
                )
                
                # Save preprocessor for later use
                joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.joblib"))
            
            logger.info(f"Preprocessed features shape: {X_scaled.shape}")
            logger.info(f"Target shape: {y_encoded.shape}")
            
            # 4. Feature engineering and selection
            logger.info("\n=== Step 3: Feature Engineering ===")
            
            # Save feature engineering results
            engineered_data_path = os.path.join(data_dir, "processed", "engineered_data.npz")
            
            if os.path.exists(engineered_data_path) and args.skip_feature_engineering:
                logger.info(f"Loading engineered features from {engineered_data_path}")
                loaded_data = np.load(engineered_data_path)
                X_selected = loaded_data['X']
                selected_feature_names = loaded_data['feature_names']
                feature_engineer = joblib.load(os.path.join(models_dir, "feature_engineer.joblib"))
            else:
                # Create feature engineer
                feature_engineer = FeatureEngineer()
                
                # Engineer new features
                X_engineered, engineered_feature_names = feature_engineer.engineer_features(
                    X_scaled, feature_names.tolist())
                
                # Select best features
                X_selected, selected_feature_names = feature_engineer.select_features(
                    X_engineered, y_encoded, engineered_feature_names, method='anova', k=args.num_features)
                
                # Save engineered data
                np.savez(
                    engineered_data_path, 
                    X=X_selected, 
                    feature_names=np.array(selected_feature_names)
                )
                
                # Save feature engineer for later use
                joblib.dump(feature_engineer, os.path.join(models_dir, "feature_engineer.joblib"))
            
            logger.info(f"Selected features shape: {X_selected.shape}")
            
            # 5. Visualize data (optional)
            logger.info("\n=== Step 4: Data Visualization ===")
            visualizer = DataVisualizer()
            
            # Create basic visualizations (if visualizer.py is implemented)
            try:
                visualizer.plot_correlation_matrix(X_selected, selected_feature_names)
                visualizer.plot_class_distribution(y_encoded)
                visualizer.plot_feature_distributions(X_selected, selected_feature_names)
            except Exception as e:
                logger.warning(f"Visualization error: {e}")
            
            # 6. Train models
            logger.info("\n=== Step 5: Model Training ===")
            model_trainer = ModelTrainer(config={
                            'sampling_ratio': 0.05,           # Use only 5% of data
                            'training_timeout_seconds': 600,  # 10 minute timeout
                            'memory_threshold_percent': 60,   # Very conservative
                            'n_jobs': 1,                      # Single thread to reduce memory
                            'fallback_to_randomized': True,
                            'n_randomized_iterations': 5      # Fewer iterations
            })
            
            # Split data
            X_train, X_test, y_train, y_test = model_trainer.split_data(
                X_selected, y_encoded, test_size=args.test_size)
            
            # Train models
            models_to_train = args.models.split(',')
            trained_models = model_trainer.train_models(
                X_train, y_train, models_to_train=models_to_train, tune_hyperparams=True)
            
            # Save models
            model_trainer.save_models(models_dir)
            
            # 7. Evaluate models
            logger.info("\n=== Step 6: Model Evaluation ===")
            evaluator = ModelEvaluator()
            
            # Evaluate each model
            evaluation_results = evaluator.evaluate_models(trained_models, X_test, y_test)
            
            # Create visualizations for evaluation
            for model_name in trained_models.keys():
                evaluator.plot_confusion_matrix(model_name)
                evaluator.print_classification_report(model_name)
            
            # Plot ROC curves for all models
            evaluator.plot_roc_curve()
            
            # 8. Show feature importance for best model
            if model_trainer.best_model_name in ['random_forest', 'gradient_boosting', 'decision_tree']:
                logger.info("\n=== Feature Importance ===")
                evaluator.feature_importance(
                    model_trainer.best_model, selected_feature_names, 
                    title=f"Feature Importance ({model_trainer.best_model_name})"
                )
            
            logger.info("\n=== Pipeline completed successfully ===")
            
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()