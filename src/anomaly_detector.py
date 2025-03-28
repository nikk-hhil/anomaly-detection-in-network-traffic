import numpy as np
import pandas as pd
import joblib
import logging
import os
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
import warnings

# Get logger
logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Class for detecting network traffic anomalies using trained models.
    """
    
    def __init__(self, model_path: str = None, preprocessor_path: str = None, 
                feature_engineer_path: str = None, threshold: float = 0.5):
        """
        Initialize AnomalyDetector with trained model and preprocessing components.
        
        Parameters:
        -----------
        model_path : str
            Path to the trained model
        preprocessor_path : str
            Path to the preprocessor
        feature_engineer_path : str
            Path to the feature engineer
        threshold : float
            Probability threshold for anomaly detection (default: 0.5)
        """
        self.model = None
        self.preprocessor = None
        self.feature_engineer = None
        self.threshold = threshold
        self.label_mapping = {}
        self.inverse_label_mapping = {}
        
        # Load components if paths are provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        if preprocessor_path and os.path.exists(preprocessor_path):
            self.load_preprocessor(preprocessor_path)
        
        if feature_engineer_path and os.path.exists(feature_engineer_path):
            self.load_feature_engineer(feature_engineer_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load trained model from file.
        
        Parameters:
        -----------
        model_path : str
            Path to the trained model
        """
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_preprocessor(self, preprocessor_path: str) -> None:
        """
        Load preprocessor from file.
        
        Parameters:
        -----------
        preprocessor_path : str
            Path to the preprocessor
        """
        try:
            self.preprocessor = joblib.load(preprocessor_path)
            logger.info(f"Loaded preprocessor from {preprocessor_path}")
            
            # Extract label mapping if available
            if hasattr(self.preprocessor, 'label_encoder') and self.preprocessor.label_encoder is not None:
                classes = self.preprocessor.label_encoder.classes_
                indices = range(len(classes))
                self.label_mapping = dict(zip(indices, classes))
                self.inverse_label_mapping = dict(zip(classes, indices))
                logger.info(f"Extracted label mapping: {self.label_mapping}")
        except Exception as e:
            logger.error(f"Error loading preprocessor: {e}")
            raise
    
    def load_feature_engineer(self, feature_engineer_path: str) -> None:
        """
        Load feature engineer from file.
        
        Parameters:
        -----------
        feature_engineer_path : str
            Path to the feature engineer
        """
        try:
            self.feature_engineer = joblib.load(feature_engineer_path)
            logger.info(f"Loaded feature engineer from {feature_engineer_path}")
        except Exception as e:
            logger.error(f"Error loading feature engineer: {e}")
            raise
    
    def set_threshold(self, threshold: float) -> None:
        """
        Set probability threshold for anomaly detection.
        
        Parameters:
        -----------
        threshold : float
            Probability threshold (0.0 to 1.0)
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        self.threshold = threshold
        logger.info(f"Set detection threshold to {threshold}")
    
    def _get_class_name(self, class_idx: int) -> str:
        """
        Get class name from class index.
        
        Parameters:
        -----------
        class_idx : int
            Class index
            
        Returns:
        --------
        str
            Class name
        """
        return self.label_mapping.get(class_idx, f"Class_{class_idx}")
    
    def preprocess_data(self, data: pd.DataFrame, target_column: Optional[str] = None) -> np.ndarray:
        """
        Preprocess input data using loaded preprocessor.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        target_column : Optional[str]
            Name of target column (if present)
            
        Returns:
        --------
        np.ndarray
            Preprocessed features
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not loaded. Call load_preprocessor() first.")
        
        # Copy data to avoid modifying original
        data_copy = data.copy()
        
        logger.info(f"Preprocessing data with shape {data.shape}")
        
        # Extract target if present
        y = None
        if target_column and target_column in data_copy.columns:
            y = data_copy[target_column].values
            
        # Clean data
        cleaned_data = self.preprocessor.clean_data(data_copy, target_column)
        
        # Encode categorical features
        encoded_data = self.preprocessor.encode_categorical(cleaned_data, target_column)
        
        # Scale features
        if target_column and target_column in encoded_data.columns:
            X_scaled, _ = self.preprocessor.scale_features(encoded_data, target_column)
        else:
            X_scaled = self.preprocessor.scaler.transform(encoded_data)
        
        logger.info(f"Preprocessing complete. Output shape: {X_scaled.shape}")
        return X_scaled
    
    def engineer_features(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """
        Engineer features using loaded feature engineer.
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
        feature_names : List[str]
            Names of input features
            
        Returns:
        --------
        np.ndarray
            Engineered features
        """
        if self.feature_engineer is None:
            raise ValueError("Feature engineer not loaded. Call load_feature_engineer() first.")
        
        logger.info(f"Engineering features from input with shape {X.shape}")
        
        # Engineer new features
        X_engineered, _ = self.feature_engineer.engineer_features(X, feature_names)
        
        # Select features (if applicable)
        if hasattr(self.feature_engineer, 'feature_selector') and self.feature_engineer.feature_selector is not None:
            X_selected = self.feature_engineer.feature_selector.transform(X_engineered)
            logger.info(f"Feature engineering complete. Output shape: {X_selected.shape}")
            return X_selected
        
        # For PCA
        elif hasattr(self.feature_engineer, 'pca_model') and self.feature_engineer.pca_model is not None:
            X_selected = self.feature_engineer.pca_model.transform(X_engineered)
            logger.info(f"Feature engineering complete. Output shape: {X_selected.shape}")
            return X_selected
        
        else:
            logger.info(f"Feature engineering complete (no selection applied). Output shape: {X_engineered.shape}")
            return X_engineered
    
    def predict(self, data: pd.DataFrame, target_column: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies in the input data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        target_column : Optional[str]
            Name of target column (if present, for metrics calculation)
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Predicted class labels and probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess data
        X_preprocessed = self.preprocess_data(data, target_column)
        
        # Get feature names (excluding target column)
        feature_names = [col for col in data.columns if col != target_column]
        
        # Engineer features
        if self.feature_engineer is not None:
            try:
                X_final = self.engineer_features(X_preprocessed, feature_names)
            except Exception as e:
                logger.warning(f"Feature engineering failed: {e}. Using preprocessed features.")
                X_final = X_preprocessed
        else:
            X_final = X_preprocessed
        
        # Make predictions
        logger.info(f"Making predictions on data with shape {X_final.shape}")
        try:
            # Get probabilities if supported
            if hasattr(self.model, 'predict_proba'):
                y_proba = self.model.predict_proba(X_final)
                
                # For binary classification, extract positive class probability
                if y_proba.shape[1] == 2:
                    y_proba_positive = y_proba[:, 1]
                else:
                    # For multiclass, max probability
                    y_proba_positive = np.max(y_proba, axis=1)
                
                # Predict class based on threshold
                y_pred = (y_proba_positive >= self.threshold).astype(int)
                
                # For multiclass, get actual predicted class
                if y_proba.shape[1] > 2:
                    y_pred = np.argmax(y_proba, axis=1)
                
                return y_pred, y_proba
            else:
                # Use predict method if predict_proba not available
                y_pred = self.model.predict(X_final)
                return y_pred, None
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def evaluate(self, data: pd.DataFrame, target_column: str) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data with target column
        target_column : str
            Name of target column
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, 
            roc_auc_score, confusion_matrix, classification_report
        )
        
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Get true labels
        y_true = data[target_column].values
        
        # Encode labels if necessary
        if self.inverse_label_mapping and y_true.dtype == object:
            y_true_encoded = np.array([self.inverse_label_mapping.get(label, -1) for label in y_true])
            if -1 in y_true_encoded:
                logger.warning("Some labels in test data were not seen during training")
            y_true = y_true_encoded
        
        # Make predictions
        y_pred, y_proba = self.predict(data, target_column)
        
        # Calculate metrics
        metrics = {}
        
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Determine if binary or multiclass
        n_classes = len(np.unique(y_true))
        
        if n_classes == 2:
            # Binary classification
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
            
            if y_proba is not None:
                try:
                    if y_proba.ndim > 1 and y_proba.shape[1] == 2:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                    else:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                except Exception as e:
                    logger.warning(f"ROC AUC calculation error: {e}")
        else:
            # Multiclass classification
            metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            if y_proba is not None:
                try:
                    metrics['roc_auc_macro'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
                except Exception as e:
                    logger.warning(f"ROC AUC calculation error: {e}")
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Log results
        logger.info("\n=== Evaluation Results ===")
        for metric, value in metrics.items():
            if metric != 'confusion_matrix':
                logger.info(f"{metric}: {value:.4f}")
        
        # Log classification report
        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(y_true, y_pred))
        
        return metrics
    
    def predict_batch(self, data_path: str, output_path: str, 
                    target_column: Optional[str] = None, batch_size: int = 10000) -> None:
        """
        Make predictions on a large dataset in batches.
        
        Parameters:
        -----------
        data_path : str
            Path to input data file (CSV)
        output_path : str
            Path to save predictions
        target_column : Optional[str]
            Name of target column (if present)
        batch_size : int
            Batch size for processing
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Input file not found: {data_path}")
        
        logger.info(f"Processing {data_path} in batches of {batch_size}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Process in batches
        for i, chunk in enumerate(pd.read_csv(data_path, chunksize=batch_size)):
            logger.info(f"Processing batch {i+1}")
            
            # Make predictions
            y_pred, y_proba = self.predict(chunk, target_column)
            
            # Create result dataframe
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
            
            # Save batch results
            mode = 'w' if i == 0 else 'a'
            header = i == 0
            result.to_csv(output_path, mode=mode, header=header, index=False)
            
            logger.info(f"Saved batch {i+1} predictions")
        
        logger.info(f"Predictions saved to {output_path}")
    
    def explain_prediction(self, data: pd.DataFrame, row_index: int,
                         target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Explain a specific prediction using feature importance.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        row_index : int
            Index of the row to explain
        target_column : Optional[str]
            Name of target column (if present)
            
        Returns:
        --------
        Dict[str, Any]
            Explanation data
        """
        try:
            import shap
        except ImportError:
            logger.error("SHAP library not installed. Install with 'pip install shap'")
            return {"error": "SHAP library not installed"}
        
        if row_index >= len(data):
            raise ValueError(f"Row index {row_index} out of bounds for data with {len(data)} rows")
        
        # Get single row
        single_row = data.iloc[[row_index]]
        
        # Preprocess
        X_preprocessed = self.preprocess_data(single_row, target_column)
        
        # Get feature names (excluding target column)
        feature_names = [col for col in data.columns if col != target_column]
        
        # Engineer features
        if self.feature_engineer is not None:
            try:
                X_final = self.engineer_features(X_preprocessed, feature_names)
                
                # Update feature names if using feature engineer
                if hasattr(self.feature_engineer, 'selected_features') and self.feature_engineer.selected_features is not None:
                    feature_names = self.feature_engineer.selected_features
                elif hasattr(self.feature_engineer, 'pca_model') and self.feature_engineer.pca_model is not None:
                    feature_names = [f"PC{i+1}" for i in range(X_final.shape[1])]
            except Exception as e:
                logger.warning(f"Feature engineering failed: {e}. Using preprocessed features.")
                X_final = X_preprocessed
        else:
            X_final = X_preprocessed
        
        # Get prediction
        if hasattr(self.model, 'predict_proba'):
            prediction = self.model.predict_proba(X_final)[0]
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class]
        else:
            predicted_class = self.model.predict(X_final)[0]
            confidence = None
        
        # Convert class to label if mapping exists
        predicted_label = self._get_class_name(predicted_class)
        
        # Create explanation based on model type
        explanation = {
            'row_index': row_index,
            'predicted_class': int(predicted_class),
            'predicted_label': predicted_label,
            'confidence': float(confidence) if confidence is not None else None,
            'feature_importance': {},
            'feature_values': {}
        }
        
        # Get original feature values
        for feature in data.columns:
            if feature != target_column:
                value = single_row[feature].values[0]
                explanation['feature_values'][feature] = value
        
        # Get feature importance based on model type
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            importances = self.model.feature_importances_
            for i, importance in enumerate(importances):
                if i < len(feature_names):
                    explanation['feature_importance'][feature_names[i]] = float(importance)
            
        elif isinstance(self.model, (LogisticRegression)):
            # Linear models
            coefs = self.model.coef_
            if coefs.shape[0] == 1:  # Binary case
                for i, coef in enumerate(coefs[0]):
                    if i < len(feature_names):
                        explanation['feature_importance'][feature_names[i]] = float(abs(coef))
            else:  # Multiclass case
                for i, coef in enumerate(coefs[predicted_class]):
                    if i < len(feature_names):
                        explanation['feature_importance'][feature_names[i]] = float(abs(coef))
        
        else:
            # Use SHAP for other models
            try:
                # Create explainer
                if hasattr(self.model, 'predict_proba'):
                    explainer = shap.Explainer(self.model.predict_proba, X_final)
                else:
                    explainer = shap.Explainer(self.model.predict, X_final)
                
                # Get SHAP values
                shap_values = explainer(X_final)
                
                # Extract values for the predicted class
                if len(shap_values.shape) > 2:  # Multiclass case
                    shap_for_class = shap_values[0, :, predicted_class]
                else:  # Binary case
                    shap_for_class = shap_values[0, :]
                
                # Add to explanation
                for i, value in enumerate(shap_for_class):
                    if i < len(feature_names):
                        explanation['feature_importance'][feature_names[i]] = float(abs(value))
            
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")
                explanation['error'] = f"SHAP explanation failed: {str(e)}"
        
        return explanation