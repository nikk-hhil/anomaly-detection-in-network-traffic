import os
import joblib
import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional, List, Any
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# Set up logger
logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Class for detecting anomalies in network traffic using trained models.
    """
    
    def __init__(self, model_path: str, preprocessor_path: str, 
                feature_engineer_path: str, threshold: float = 0.5,
                output_dir: str = './results'):
        """
        Initialize the anomaly detector.
        
        Parameters:
        -----------
        model_path : str
            Path to the trained model file
        preprocessor_path : str
            Path to the preprocessor file
        feature_engineer_path : str
            Path to the feature engineer file
        threshold : float
            Probability threshold for anomaly detection
        output_dir : str
            Directory to save results
        """
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.feature_engineer_path = feature_engineer_path
        self.threshold = threshold
        self.output_dir = output_dir
        
        # Load components
        self.model = None
        self.preprocessor = None
        self.feature_engineer = None
        
        self.label_mapping = None
        self.inverse_label_mapping = None
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the model and preprocessing components
        self.load_model(model_path)
        self.load_preprocessor(preprocessor_path)
        self.load_feature_engineer(feature_engineer_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load the trained model.
        
        Parameters:
        -----------
        model_path : str
            Path to the model file
        """
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_preprocessor(self, preprocessor_path: str) -> None:
        """
        Load the preprocessor.
        
        Parameters:
        -----------
        preprocessor_path : str
            Path to the preprocessor file
        """
        try:
            self.preprocessor = joblib.load(preprocessor_path)
            logger.info(f"Loaded preprocessor from {preprocessor_path}")
            
            # Extract label mapping from preprocessor if available
            if hasattr(self.preprocessor, 'label_encoder') and self.preprocessor.label_encoder is not None:
                self.label_mapping = {i: label for i, label in enumerate(self.preprocessor.label_encoder.classes_)}
                self.inverse_label_mapping = {label: i for i, label in self.label_mapping.items()}
                logger.info(f"Extracted label mapping: {self.label_mapping}")
        except Exception as e:
            logger.error(f"Error loading preprocessor: {e}")
            raise
    
    def load_feature_engineer(self, feature_engineer_path: str) -> None:
        """
        Load the feature engineer.
        
        Parameters:
        -----------
        feature_engineer_path : str
            Path to the feature engineer file
        """
        try:
            self.feature_engineer = joblib.load(feature_engineer_path)
            logger.info(f"Loaded feature engineer from {feature_engineer_path}")
        except Exception as e:
            logger.error(f"Error loading feature engineer: {e}")
            raise
    
    def _get_class_name(self, class_idx: int) -> str:
        """
        Get the class name for a given class index.
        
        Parameters:
        -----------
        class_idx : int
            Class index
            
        Returns:
        --------
        str
            Class name
        """
        if self.label_mapping and class_idx in self.label_mapping:
            return self.label_mapping[class_idx]
        return f"Class_{class_idx}"
    
    def set_threshold(self, threshold: float) -> None:
        """
        Set the detection threshold.
        
        Parameters:
        -----------
        threshold : float
            New threshold value
        """
        self.threshold = threshold
        logger.info(f"Set detection threshold to {threshold}")
    
    def preprocess_data(self, data: pd.DataFrame, target_column: Optional[str] = None) -> np.ndarray:
        """
        Preprocess data for prediction.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        target_column : str, optional
            Target column name (if present)
            
        Returns:
        --------
        np.ndarray
            Preprocessed data
        """
        try:
            # Copy data to avoid modifying the original
            data_copy = data.copy()
            
            # Step 1: Handle missing values and clean the data
            if target_column and target_column in data_copy.columns:
                cleaned_data = self.preprocessor.clean_data(data_copy, target_column)
                # Step 2: Encode categorical features
                encoded_data = self.preprocessor.encode_categorical(cleaned_data, target_column)
                # Step 3: Scale features
                X_scaled, _ = self.preprocessor.scale_features(encoded_data, target_column)
                feature_names = encoded_data.drop(columns=[target_column]).columns.tolist()
            else:
                # If no target column, we need to manually apply similar preprocessing
                # This assumes the preprocessor has the necessary attributes from training
                
                # Handle missing values
                numeric_cols = data_copy.select_dtypes(include=['int64', 'float64']).columns
                for col in numeric_cols:
                    data_copy[col] = data_copy[col].fillna(data_copy[col].median())
                    data_copy[col] = data_copy[col].replace([np.inf, -np.inf], data_copy[col].median())
                
                # Encode categorical columns if any
                cat_cols = data_copy.select_dtypes(include=['object']).columns
                for col in cat_cols:
                    if hasattr(self.preprocessor, 'feature_encoders') and col in self.preprocessor.feature_encoders:
                        # Handle unseen categories
                        encoder = self.preprocessor.feature_encoders[col]
                        unseen_cats = set(data_copy[col].unique()) - set(encoder.classes_)
                        if unseen_cats:
                            for cat in unseen_cats:
                                data_copy.loc[data_copy[col] == cat, col] = encoder.classes_[0]
                        data_copy[col] = encoder.transform(data_copy[col])
                
                # Scale numeric features
                if hasattr(self.preprocessor, 'scaler') and self.preprocessor.scaler is not None:
                    X_scaled = self.preprocessor.scaler.transform(data_copy)
                else:
                    X_scaled = data_copy.values
                
                feature_names = data_copy.columns.tolist()
            
            # Step 4: Apply feature engineering and selection
            if self.feature_engineer is not None:
                # First apply feature engineering
                X_engineered, engineered_feature_names = self.feature_engineer.engineer_features(
                    X_scaled, feature_names)
                
                # Then apply feature selection to get the final set of features
                X_selected = self.feature_engineer.transform_feature_selection(X_engineered)  # Assuming 20 features were selected during training
                
                return X_selected
            else:
                # If no feature engineer is available, return the scaled features
                # This might cause dimension mismatch with the model
                logger.warning("No feature engineer available. Using scaled features only.")
                return X_scaled
                
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise
    
    def predict(self, data: pd.DataFrame, target_column: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        target_column : str, optional
            Target column name (if present)
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Predicted labels and probabilities
        """
        try:
            # Preprocess data
            X_final = self.preprocess_data(data, target_column)
            
            logger.info(f"Preprocessed data shape: {X_final.shape}")
            
            # Make predictions
            y_pred = self.model.predict(X_final)
            
            # Get probabilities if possible
            try:
                y_proba = self.model.predict_proba(X_final)
            except Exception as e:
                logger.warning(f"Could not get prediction probabilities: {e}")
                y_proba = None
            
            return y_pred, y_proba
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def evaluate(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Evaluate model performance on labeled data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data with target column
        target_column : str
            Target column name
            
        Returns:
        --------
        Dict[str, Any]
            Evaluation metrics
        """
        try:
            # Get true labels
            y_true = data[target_column].values
            
            # Encode labels if necessary
            if y_true.dtype == object and self.preprocessor.label_encoder is not None:
                y_true = self.preprocessor.label_encoder.transform(y_true)
            
            # Make predictions
            y_pred, y_proba = self.predict(data, target_column)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1_score': f1_score(y_true, y_pred, average='weighted'),
                'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
                'classification_report': classification_report(y_true, y_pred, output_dict=True)
            }
            
            # Log metrics
            logger.info(f"Evaluation metrics:")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Precision: {metrics['precision']:.4f}")
            logger.info(f"Recall: {metrics['recall']:.4f}")
            logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            raise
    
    def predict_batch(self, input_file: str, output_file: str, 
                     target_column: Optional[str] = None,
                     batch_size: int = 10000) -> None:
        """
        Process a large file in batches.
        
        Parameters:
        -----------
        input_file : str
            Path to input CSV file
        output_file : str
            Path to output CSV file
        target_column : str, optional
            Target column name (if present)
        batch_size : int
            Batch size for processing
        """
        try:
            # Get total rows for progress reporting
            total_rows = sum(1 for _ in open(input_file)) - 1  # Subtract header row
            logger.info(f"Processing {total_rows} rows in batches of {batch_size}")
            
            # Process in batches
            chunk_iter = pd.read_csv(input_file, chunksize=batch_size)
            
            # Process first chunk to get column structure for output
            first_chunk = next(chunk_iter)
            y_pred, y_proba = self.predict(first_chunk, target_column)
            
            # Create result DataFrame for first chunk
            result = first_chunk.copy()
            result['predicted_class'] = y_pred
            
            # Add class names if available
            if self.label_mapping:
                result['predicted_label'] = [self._get_class_name(pred) for pred in y_pred]
            
            # Add probabilities if available
            if y_proba is not None:
                if y_proba.ndim > 1:
                    # Multi-class case
                    for j in range(y_proba.shape[1]):
                        result[f'probability_class_{j}'] = y_proba[:, j]
                        if self.label_mapping:
                            class_name = self._get_class_name(j)
                            result[f'probability_{class_name}'] = y_proba[:, j]
                else:
                    # Binary case
                    result['probability'] = y_proba
            
            # Write first chunk with header
            result.to_csv(output_file, index=False, mode='w')
            
            # Process remaining chunks
            processed_rows = len(first_chunk)
            logger.info(f"Processed {processed_rows}/{total_rows} rows ({processed_rows/total_rows:.1%})")
            
            for i, chunk in enumerate(chunk_iter, 1):
                y_pred, y_proba = self.predict(chunk, target_column)
                
                # Create result DataFrame
                result = chunk.copy()
                result['predicted_class'] = y_pred
                
                # Add class names if available
                if self.label_mapping:
                    result['predicted_label'] = [self._get_class_name(pred) for pred in y_pred]
                
                # Add probabilities if available
                if y_proba is not None:
                    if y_proba.ndim > 1:
                        # Multi-class case
                        for j in range(y_proba.shape[1]):
                            result[f'probability_class_{j}'] = y_proba[:, j]
                            if self.label_mapping:
                                class_name = self._get_class_name(j)
                                result[f'probability_{class_name}'] = y_proba[:, j]
                    else:
                        # Binary case
                        result['probability'] = y_proba
                
                # Append to output file without header
                result.to_csv(output_file, index=False, mode='a', header=False)
                
                # Update progress
                processed_rows += len(chunk)
                logger.info(f"Processed {processed_rows}/{total_rows} rows ({processed_rows/total_rows:.1%})")
            
            logger.info(f"Batch processing complete. Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            raise
    
    def explain_prediction(self, data: pd.DataFrame, row_idx: int,
                          target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate an explanation for a specific prediction.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        row_idx : int
            Index of the row to explain
        target_column : str, optional
            Target column name (if present)
            
        Returns:
        --------
        Dict[str, Any]
            Explanation details
        """
        try:
            # Get the row to explain
            row_data = data.iloc[[row_idx]].copy()
            
            # Get prediction
            y_pred, y_proba = self.predict(row_data, target_column)
            
            # Get predicted class and name
            pred_class = int(y_pred[0])
            pred_class_name = self._get_class_name(pred_class)
            
            # Base explanation with key data and prediction
            explanation = {
                'row_idx': int(row_idx),
                'prediction': {
                    'class': int(pred_class),
                    'class_name': pred_class_name,
                    'confidence': float(y_proba[0, pred_class]) if y_proba is not None and y_proba.ndim > 1 else None
                },
                'input_data': row_data.to_dict('records')[0],
            }
            
            # Add true label if available
            if target_column and target_column in row_data.columns:
                true_label = row_data[target_column].iloc[0]
                true_class = self.inverse_label_mapping.get(true_label, -1) if self.inverse_label_mapping else -1
                explanation['true_label'] = {
                    'class': int(true_class) if true_class != -1 else None,
                    'class_name': str(true_label)
                }
            
            # Try to add feature importance if the model supports it
            if hasattr(self.model, 'feature_importances_'):
                # Preprocess the data to get the features used by the model
                X_final = self.preprocess_data(row_data, target_column)
                
                # Get feature names if available
                if hasattr(self.feature_engineer, 'selected_features_') and self.feature_engineer.selected_features_ is not None:
                    feature_names = self.feature_engineer.selected_features_
                else:
                    feature_names = [f"feature_{i}" for i in range(X_final.shape[1])]
                
                # Get feature importances
                importances = self.model.feature_importances_
                
                # Add to explanation
                explanation['feature_importances'] = [
                    {'name': name, 'importance': float(imp)} 
                    for name, imp in zip(feature_names, importances)
                ]
                
                # Sort by importance
                explanation['feature_importances'].sort(key=lambda x: x['importance'], reverse=True)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining prediction: {e}")
            raise