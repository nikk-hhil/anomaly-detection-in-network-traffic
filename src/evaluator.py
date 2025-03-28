import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    classification_report
)

class ModelEvaluator:
    """
    Class for evaluating anomaly detection models.
    """
    
    def __init__(self):
        self.results = {}
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str = "Model") -> Dict[str, float]:
        """
        Evaluate a single model and return performance metrics.
        
        Parameters:
        -----------
        model : Any
            Trained model with predict and predict_proba methods
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test target
        model_name : str
            Name of the model for reporting
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of performance metrics
        """
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Get probability predictions if available
        try:
            y_prob = model.predict_proba(X_test)
            # For binary classification, we need the probability of the positive class
            if y_prob.shape[1] == 2:
                y_prob = y_prob[:, 1]
        except (AttributeError, NotImplementedError):
            y_prob = None
        
        # Calculate metrics
        metrics = {}
        
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        
        # Binary classification
        if len(np.unique(y_test)) == 2:
            metrics['precision'] = precision_score(y_test, y_pred)
            metrics['recall'] = recall_score(y_test, y_pred)
            metrics['f1'] = f1_score(y_test, y_pred)
            
            if y_prob is not None:
                metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
        # Multi-class classification
        else:
            metrics['precision_macro'] = precision_score(y_test, y_pred, average='macro')
            metrics['recall_macro'] = recall_score(y_test, y_pred, average='macro')
            metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro')
            
            if y_prob is not None:
                try:
                    metrics['roc_auc_macro'] = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
                except Exception:
                    # Skip ROC AUC if there's an issue (e.g., with predict_proba)
                    pass
        
        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
        # Print results
        print(f"\n=== {model_name} Performance ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def evaluate_models(self, models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate multiple models and compare their performance.
        
        Parameters:
        -----------
        models : Dict[str, Any]
            Dictionary of trained models
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test target
            
        Returns:
        --------
        Dict[str, Dict[str, float]]
            Dictionary of model names and their performance metrics
        """
        all_metrics = {}
        
        for model_name, model in models.items():
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            all_metrics[model_name] = metrics
        
        return all_metrics
    
    def plot_confusion_matrix(self, model_name: str, normalize: bool = True) -> None:
        """
        Plot confusion matrix for a specific model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to plot
        normalize : bool
            Whether to normalize the confusion matrix
        """
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not found in evaluation results")
        
        y_test = self.results[model_name]['y_test']
        y_pred = self.results[model_name]['y_pred']
        
        # Get confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = f'Normalized Confusion Matrix - {model_name}'
        else:
            fmt = 'd'
            title = f'Confusion Matrix - {model_name}'
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", cbar=False)
        
        # Get class labels
        classes = np.unique(np.concatenate((y_test, y_pred)))
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks + 0.5, classes)
        plt.yticks(tick_marks + 0.5, classes)
        
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, model_names: List[str] = None) -> None:
        """
        Plot ROC curves for one or more models.
        
        Parameters:
        -----------
        model_names : List[str]
            Names of models to include in the plot
        """
        if model_names is None:
            model_names = list(self.results.keys())
        
        plt.figure(figsize=(10, 8))
        
        for model_name in model_names:
            if model_name not in self.results:
                print(f"Warning: Model '{model_name}' not found in results. Skipping.")
                continue
            
            y_test = self.results[model_name]['y_test']
            y_prob = self.results[model_name]['y_prob']
            
            if y_prob is None:
                print(f"Warning: No probability predictions for '{model_name}'. Skipping.")
                continue
            
            # Check if binary classification
            if len(np.unique(y_test)) == 2:
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = self.results[model_name]['metrics'].get('roc_auc', 0)
                
                # Plot
                plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
            else:
                print(f"Warning: ROC curve is only supported for binary classification. '{model_name}' has {len(np.unique(y_test))} classes.")
        
        # Add diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
    
    def print_classification_report(self, model_name: str) -> None:
        """
        Print detailed classification report for a specific model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        """
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not found in evaluation results")
        
        y_test = self.results[model_name]['y_test']
        y_pred = self.results[model_name]['y_pred']
        
        print(f"\n=== Classification Report for {model_name} ===")
        print(classification_report(y_test, y_pred))
    
    def feature_importance(self, model, feature_names: List[str], 
                          title: str = "Feature Importance", top_n: int = 20) -> None:
        """
        Plot feature importance for supported models.
        
        Parameters:
        -----------
        model : Any
            Trained model with feature_importances_ attribute
        feature_names : List[str]
            Names of features
        title : str
            Title for the plot
        top_n : int
            Number of top features to display
        """
        # Check if model supports feature importance
        if not hasattr(model, 'feature_importances_'):
            print("Model does not support feature importance visualization")
            return
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort features by importance
        sorted_idx = np.argsort(importances)[::-1]
        
        # Select top N features
        top_n = min(top_n, len(feature_names))
        top_idx = sorted_idx[:top_n]
        top_features = [feature_names[i] for i in top_idx]
        top_importances = importances[top_idx]
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.barh(range(top_n), top_importances, align='center')
        plt.yticks(range(top_n), top_features)
        plt.title(title)
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()