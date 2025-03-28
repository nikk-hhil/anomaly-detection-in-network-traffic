import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataVisualizer:
    """
    Class for creating visualizations of network traffic data and model results.
    """
    
    def __init__(self, output_dir: str = './visualizations'):
        """
        Initialize the visualizer with an output directory.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save visualizations
        """
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_correlation_matrix(self, X: np.ndarray, feature_names: List[str], 
                               save_fig: bool = True) -> None:
        """
        Plot correlation matrix of features.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        feature_names : List[str]
            Names of features
        save_fig : bool
            Whether to save the figure to disk
        """
        # Convert to DataFrame for easier correlation calculation
        X_df = pd.DataFrame(X, columns=feature_names)
        
        # Calculate correlation matrix
        corr_matrix = X_df.corr()
        
        # Plot
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Use seaborn heatmap
        sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm", annot=False,
                   vmin=-1, vmax=1, square=True, linewidths=.5)
        
        plt.title("Feature Correlation Matrix", fontsize=16)
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(f"{self.output_dir}/correlation_matrix.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_class_distribution(self, y: np.ndarray, save_fig: bool = True) -> None:
        """
        Plot distribution of target classes.
        
        Parameters:
        -----------
        y : np.ndarray
            Target array
        save_fig : bool
            Whether to save the figure to disk
        """
        plt.figure(figsize=(10, 6))
        
        # Count class frequencies
        unique_classes, counts = np.unique(y, return_counts=True)
        
        # Create bar plot
        bars = plt.bar(unique_classes, counts)
        
        # Add count labels on top of bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count} ({count/len(y):.1%})', 
                    ha='center', va='bottom', rotation=0)
        
        plt.title("Class Distribution", fontsize=16)
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.xticks(unique_classes)
        plt.grid(axis='y', alpha=0.3)
        
        if save_fig:
            plt.savefig(f"{self.output_dir}/class_distribution.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_distributions(self, X: np.ndarray, feature_names: List[str], 
                                 top_n: int = 6, save_fig: bool = True) -> None:
        """
        Plot distributions of top features.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        feature_names : List[str]
            Names of features
        top_n : int
            Number of top features to plot
        save_fig : bool
            Whether to save the figure to disk
        """
        # Select top N features (or all if less than top_n)
        n_features = min(top_n, X.shape[1])
        
        # Create subplot grid
        rows = int(np.ceil(n_features / 2))
        plt.figure(figsize=(14, rows * 4))
        
        for i in range(n_features):
            plt.subplot(rows, 2, i + 1)
            
            # Plot histogram with KDE
            sns.histplot(X[:, i], kde=True)
            
            plt.title(f"Distribution of {feature_names[i]}")
            plt.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(f"{self.output_dir}/feature_distributions.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_pca_variance(self, pca, save_fig: bool = True) -> None:
        """
        Plot explained variance ratio for PCA components.
        
        Parameters:
        -----------
        pca : sklearn.decomposition.PCA
            Fitted PCA model
        save_fig : bool
            Whether to save the figure to disk
        """
        plt.figure(figsize=(10, 6))
        
        # Cumulative explained variance
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        
        # Plot component variances
        plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
               pca.explained_variance_ratio_, alpha=0.6, label='Individual')
        
        # Plot cumulative variance
        plt.step(range(1, len(cum_var) + 1), cum_var, where='mid', 
                label=f'Cumulative', color='red')
        
        # Add 95% variance line
        plt.axhline(y=0.95, color='k', linestyle='-', alpha=0.7, label='95% Variance')
        
        # Get number of components for 95% variance
        n_components_95 = np.argmax(cum_var >= 0.95) + 1
        plt.scatter(n_components_95, cum_var[n_components_95-1], s=100, color='green', 
                   label=f'{n_components_95} Components: {cum_var[n_components_95-1]:.2%}')
        
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Explained Variance')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        
        if save_fig:
            plt.savefig(f"{self.output_dir}/pca_variance.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance_comparison(self, feature_importances: Dict[str, np.ndarray], 
                                          feature_names: List[str], top_n: int = 15,
                                          save_fig: bool = True) -> None:
        """
        Plot and compare feature importances from multiple models.
        
        Parameters:
        -----------
        feature_importances : Dict[str, np.ndarray]
            Dictionary of model names and their feature importance arrays
        feature_names : List[str]
            Names of features
        top_n : int
            Number of top features to show
        save_fig : bool
            Whether to save the figure to disk
        """
        # Determine number of models
        n_models = len(feature_importances)
        
        if n_models == 0:
            logger.warning("No feature importances provided")
            return
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Calculate average importance across models
        avg_importance = np.zeros(len(feature_names))
        for importance in feature_importances.values():
            # Normalize to sum to 1 if not already
            normalized = importance / importance.sum() if importance.sum() > 0 else importance
            avg_importance += normalized
        
        avg_importance /= n_models
        
        # Sort features by average importance
        sorted_idx = np.argsort(avg_importance)[::-1]
        top_idx = sorted_idx[:top_n]
        
        # Get top feature names
        top_features = [feature_names[i] for i in top_idx]
        
        # Plot
        x = np.arange(len(top_features))
        width = 0.7 / n_models
        
        # Plot each model's feature importance
        for i, (model_name, importance) in enumerate(feature_importances.items()):
            # Normalize importance
            normalized = importance / importance.sum() if importance.sum() > 0 else importance
            # Get values for top features
            model_values = [normalized[idx] for idx in top_idx]
            plt.bar(x + (i - n_models/2 + 0.5) * width, model_values, width, label=model_name)
        
        plt.xlabel('Feature')
        plt.ylabel('Normalized Importance')
        plt.title('Feature Importance Comparison')
        plt.xticks(x, top_features, rotation=90)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(f"{self.output_dir}/feature_importance_comparison.png", 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_tsne_visualization(self, X: np.ndarray, y: np.ndarray, 
                              random_state: int = 42, save_fig: bool = True) -> None:
        """
        Create t-SNE visualization of the data.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        random_state : int
            Random seed for reproducibility
        save_fig : bool
            Whether to save the figure to disk
        """
        try:
            from sklearn.manifold import TSNE
            
            # Create t-SNE model
            tsne = TSNE(n_components=2, random_state=random_state, perplexity=30)
            
            # Sample data if too large (t-SNE is computationally expensive)
            max_samples = 5000
            if X.shape[0] > max_samples:
                logger.info(f"Sampling {max_samples} points for t-SNE visualization")
                indices = np.random.choice(X.shape[0], max_samples, replace=False)
                X_sample = X[indices]
                y_sample = y[indices]
            else:
                X_sample = X
                y_sample = y
            
            # Fit and transform
            X_tsne = tsne.fit_transform(X_sample)
            
            # Plot
            plt.figure(figsize=(12, 10))
            
            # Get unique classes
            unique_classes = np.unique(y_sample)
            
            # Create scatter plot for each class
            for class_value in unique_classes:
                mask = y_sample == class_value
                plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=f'Class {class_value}', 
                         alpha=0.7, edgecolors='w', linewidth=0.5)
            
            plt.title('t-SNE Visualization')
            plt.xlabel('t-SNE Feature 1')
            plt.ylabel('t-SNE Feature 2')
            plt.legend()
            plt.grid(alpha=0.3)
            
            if save_fig:
                plt.savefig(f"{self.output_dir}/tsne_visualization.png", dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error in t-SNE visualization: {e}")