import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Get logger
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Class for feature engineering and selection in network traffic anomaly detection.
    """
    
    def __init__(self):
        self.feature_selector = None
        self.selected_features = None
        self.pca_model = None
    
    def engineer_features(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Engineer new features based on domain knowledge of network traffic.
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
        feature_names : List[str]
            Names of original features
            
        Returns:
        --------
        Tuple[np.ndarray, List[str]]
            Features with engineered features added and updated feature names
        """
        logger.info("\n=== Engineering New Features ===")
        
        # Convert to DataFrame for easier feature engineering
        X_df = pd.DataFrame(X, columns=feature_names)
        
        # Create new feature names list
        new_feature_names = feature_names.copy()
        
        # Check if flow duration and packet-related features exist to create rate-based features
        flow_features = [col for col in X_df.columns if 'Flow' in col and 'Duration' in col]
        packet_features = [col for col in X_df.columns if 'Packet' in col or 'Pkts' in col]
        byte_features = [col for col in X_df.columns if 'Byte' in col or 'Byt' in col]
        
        # Engineer rate-based features
        if flow_features and packet_features:
            flow_duration_col = flow_features[0]
            
            # Avoid division by zero
            X_df['Flow_Duration_Safe'] = X_df[flow_duration_col].copy()
            X_df.loc[X_df['Flow_Duration_Safe'] == 0, 'Flow_Duration_Safe'] = 1
            
            # Calculate packet rate features
            for packet_feature in packet_features:
                rate_feature_name = f"{packet_feature}_Rate"
                X_df[rate_feature_name] = X_df[packet_feature] / X_df['Flow_Duration_Safe']
                new_feature_names.append(rate_feature_name)
                logger.info(f"Created rate feature: {rate_feature_name}")
            
            # Calculate byte rate features
            for byte_feature in byte_features:
                rate_feature_name = f"{byte_feature}_Rate"
                X_df[rate_feature_name] = X_df[byte_feature] / X_df['Flow_Duration_Safe']
                new_feature_names.append(rate_feature_name)
                logger.info(f"Created rate feature: {rate_feature_name}")
            
            # Drop temporary column
            X_df = X_df.drop(columns=['Flow_Duration_Safe'])
        
        # Detect and create ratio features between forward and backward traffic
        fwd_features = [col for col in X_df.columns if 'Fwd' in col]
        bwd_features = [col for col in X_df.columns if 'Bwd' in col]
        
        # Match corresponding forward and backward features
        for fwd_feature in fwd_features:
            # Try to find matching backward feature by replacing 'Fwd' with 'Bwd'
            bwd_feature = fwd_feature.replace('Fwd', 'Bwd')
            if bwd_feature in bwd_features:
                ratio_feature_name = f"{fwd_feature.replace('Fwd', '')}_Ratio"
                # Avoid division by zero
                denominator = X_df[bwd_feature].copy()
                denominator[denominator == 0] = 1
                X_df[ratio_feature_name] = X_df[fwd_feature] / denominator
                new_feature_names.append(ratio_feature_name)
                logger.info(f"Created ratio feature: {ratio_feature_name}")
        
        # Create statistical features based on time windows (if applicable)
        time_features = [col for col in X_df.columns if 'Time' in col]
        if time_features:
            # Engineering time-based features would depend on the dataset structure
            # This is a placeholder for time-based feature engineering
            logger.info("Time-based features found but not engineered in this version")
        
        # Create entropy-based features for identifying randomness patterns
        # Common in encrypted attack traffic, DNS tunneling, etc.
        text_features = [col for col in X_df.columns if X_df[col].dtype == 'object']
        for text_feature in text_features:
            try:
                # Calculate Shannon entropy for text features
                entropy_feature_name = f"{text_feature}_Entropy"
                
                # Define entropy calculation function
                def calc_entropy(text):
                    if not isinstance(text, str) or len(text) == 0:
                        return 0
                    probs = [text.count(c)/len(text) for c in set(text)]
                    entropy = -sum(p * np.log2(p) for p in probs)
                    return entropy
                
                # Apply entropy calculation
                X_df[entropy_feature_name] = X_df[text_feature].apply(calc_entropy)
                new_feature_names.append(entropy_feature_name)
                logger.info(f"Created entropy feature: {entropy_feature_name}")
            except Exception as e:
                logger.warning(f"Error creating entropy feature for {text_feature}: {e}")
        
        # Handle protocol-based features (if applicable)
        protocol_features = [col for col in X_df.columns if 'Protocol' in col or 'proto' in col.lower()]
        if protocol_features:
            try:
                for protocol_feature in protocol_features:
                    # One-hot encode protocol if it's categorical
                    if X_df[protocol_feature].dtype == 'object' or X_df[protocol_feature].nunique() < 10:
                        # Get dummies for protocols
                        protocol_dummies = pd.get_dummies(X_df[protocol_feature], prefix=protocol_feature)
                        
                        # Add to dataframe
                        X_df = pd.concat([X_df, protocol_dummies], axis=1)
                        
                        # Add new feature names
                        new_feature_names.extend(protocol_dummies.columns.tolist())
                        logger.info(f"Created one-hot encoding for {protocol_feature} with {protocol_dummies.shape[1]} categories")
            except Exception as e:
                logger.warning(f"Error processing protocol features: {e}")
        
        # Create interaction features for numeric columns
        numeric_cols = X_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if len(numeric_cols) > 1:
            # Sample a small subset of potential interactions to avoid explosion
            import random
            if len(numeric_cols) > 10:
                # Take a random sample of columns for interactions if there are many
                sampled_cols = random.sample(numeric_cols, 10)
            else:
                sampled_cols = numeric_cols
            
            # Create meaningful interactions
            for i, col1 in enumerate(sampled_cols[:-1]):
                for col2 in sampled_cols[i+1:]:
                    # Only consider features that might be related based on names
                    # This is a simplistic approach; domain knowledge would improve this
                    if any(term in col1 and term in col2 for term in ['Packet', 'Byte', 'Flow', 'Fwd', 'Bwd']):
                        interaction_name = f"{col1}_x_{col2}"
                        X_df[interaction_name] = X_df[col1] * X_df[col2]
                        new_feature_names.append(interaction_name)
                        logger.info(f"Created interaction feature: {interaction_name}")
        
        # Convert back to numpy array
        X_engineered = X_df.values
        
        logger.info(f"Added {len(new_feature_names) - len(feature_names)} new engineered features")
        return X_engineered, new_feature_names
    
    def select_features(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                        method: str = 'anova', k: int = 20) -> Tuple[np.ndarray, List[str]]:
        """
        Select the most relevant features for anomaly detection.
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
        y : np.ndarray
            Target values
        feature_names : List[str]
            Names of features
        method : str
            Feature selection method ('anova', 'mutual_info', 'pca', or 'combined')
        k : int
            Number of features to select
            
        Returns:
        --------
        Tuple[np.ndarray, List[str]]
            Selected features and their names
        """
        logger.info(f"\n=== Selecting {k} Best Features using {method.upper()} ===")
        
        # Adjust k if it's larger than the number of features
        k = min(k, X.shape[1])
        
        if method.lower() == 'anova':
            self.feature_selector = SelectKBest(f_classif, k=k)
            X_selected = self.feature_selector.fit_transform(X, y)
            
            # Get selected feature indices and names
            selected_indices = self.feature_selector.get_support(indices=True)
            selected_feature_names = [feature_names[i] for i in selected_indices]
            selected_scores = self.feature_selector.scores_[selected_indices]
            
            # Print top features
            self._print_top_features(selected_feature_names, selected_scores)
            
        elif method.lower() == 'mutual_info':
            self.feature_selector = SelectKBest(mutual_info_classif, k=k)
            X_selected = self.feature_selector.fit_transform(X, y)
            
            # Get selected feature indices and names
            selected_indices = self.feature_selector.get_support(indices=True)
            selected_feature_names = [feature_names[i] for i in selected_indices]
            selected_scores = self.feature_selector.scores_[selected_indices]
            
            # Print top features
            self._print_top_features(selected_feature_names, selected_scores)
            
        elif method.lower() == 'pca':
            # PCA doesn't select features; it creates new components
            self.pca_model = PCA(n_components=k)
            X_selected = self.pca_model.fit_transform(X)
            
            # Create generic component names
            selected_feature_names = [f"PC{i+1}" for i in range(k)]
            
            # Print explained variance
            total_variance = sum(self.pca_model.explained_variance_ratio_)
            logger.info(f"PCA with {k} components explains {total_variance:.2%} of variance")
            
            # Print top component contributions
            self._print_pca_components(feature_names)
            
        elif method.lower() == 'combined':
            # First use ANOVA to get top 2*k features
            anova_selector = SelectKBest(f_classif, k=min(2*k, X.shape[1]))
            X_anova = anova_selector.fit_transform(X, y)
            anova_indices = anova_selector.get_support(indices=True)
            anova_features = [feature_names[i] for i in anova_indices]
            
            # Then use mutual information to select final k features from the ANOVA selected features
            mi_selector = SelectKBest(mutual_info_classif, k=k)
            X_selected = mi_selector.fit_transform(X_anova, y)
            
            # Get final selected feature indices and names
            mi_indices = mi_selector.get_support(indices=True)
            selected_feature_names = [anova_features[i] for i in mi_indices]
            
            # Store the combined selector (for consistency with interface)
            self.feature_selector = mi_selector
            
            # Log the process
            logger.info(f"Combined selection: ANOVA -> {len(anova_features)} features, then Mutual Info -> {len(selected_feature_names)} features")
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        self.selected_features = selected_feature_names
        
        return X_selected, selected_feature_names
    
    def _print_top_features(self, feature_names: List[str], scores: np.ndarray) -> None:
        """
        Print top features and their scores.
        
        Parameters:
        -----------
        feature_names : List[str]
            Names of features
        scores : np.ndarray
            Feature importance scores
        """
        # Sort features by score
        sorted_idx = np.argsort(scores)[::-1]
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_scores = scores[sorted_idx]
        
        logger.info("\nTop features and their scores:")
        for name, score in zip(sorted_features, sorted_scores):
            logger.info(f"{name}: {score:.4f}")
    
    def _print_pca_components(self, feature_names: List[str], top_n: int = 5) -> None:
        """
        Print PCA component compositions.
        
        Parameters:
        -----------
        feature_names : List[str]
            Names of original features
        top_n : int
            Number of top contributing features to show per component
        """
        if self.pca_model is None:
            logger.warning("PCA model not fitted yet")
            return
        
        logger.info("\nPCA Component Compositions:")
        
        for i, component in enumerate(self.pca_model.components_):
            # Get absolute feature contributions to this component
            contributions = np.abs(component)
            
            # Get top contributing features
            top_indices = np.argsort(contributions)[::-1][:top_n]
            top_features = [feature_names[idx] for idx in top_indices]
            top_values = [component[idx] for idx in top_indices]
            
            # Print component info
            variance_explained = self.pca_model.explained_variance_ratio_[i]
            logger.info(f"\nPC{i+1} (Explains {variance_explained:.2%} of variance)")
            
            for feature, value in zip(top_features, top_values):
                logger.info(f"  {feature}: {value:.4f}")
    
    def visualize_feature_importance(self, feature_names: List[str], 
                                     scores: np.ndarray, 
                                     title: str = "Feature Importance",
                                     top_n: int = 20) -> None:
        """
        Visualize feature importance scores.
        
        Parameters:
        -----------
        feature_names : List[str]
            Names of features
        scores : np.ndarray
            Importance scores for features
        title : str
            Title for the plot
        top_n : int
            Number of top features to show
        """
        # Sort features by importance
        indices = np.argsort(scores)[::-1]
        sorted_names = [feature_names[i] for i in indices]
        sorted_scores = scores[indices]
        
        # Limit to top_n features
        n_features = min(top_n, len(feature_names))
        top_names = sorted_names[:n_features]
        top_scores = sorted_scores[:n_features]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        plt.barh(range(n_features), top_scores, align='center')
        plt.yticks(range(n_features), top_names)
        plt.xlabel('Score')
        plt.ylabel('Features')
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def transform_feature_selection(self, X: np.ndarray) -> np.ndarray:
        """
    Apply previously fitted feature selection to new data.
    
    Parameters:
    -----------
    X : np.ndarray
        Input features
        
    Returns:
    --------
    np.ndarray
        Transformed features with only selected features
    """
        if self.feature_selector is None:
            raise ValueError("Feature selector has not been fitted yet. Call select_features first.")
    
    # Handle dimension mismatch - log information about the issue
        expected_features = getattr(self.feature_selector, 'n_features_in_', None)
        actual_features = X.shape[1]
    
        if expected_features is not None and expected_features != actual_features:
            logger.warning(f"Feature count mismatch: SelectKBest expects {expected_features} features but got {actual_features}.")
        
        # Check if we have stored selected indices
            if hasattr(self, 'selected_features_') and self.selected_features_ is not None:
            # Just return the features we know were selected, by name
            # This won't work here, as we don't have the names, just indices
                pass
        
        # As a fallback, simply select the top k features by variance
            from sklearn.feature_selection import VarianceThreshold
            k = getattr(self.feature_selector, 'k', 20)  # Default to 20 if k is not found
            selector = VarianceThreshold()
            X_var = selector.fit_transform(X)
        
        # If still too many features, just take the first k
            if X_var.shape[1] > k:
                logger.warning(f"Falling back to selecting top {k} features by variance")
            # Get indices of features with highest variance
                var = np.var(X, axis=0)
                top_indices = np.argsort(var)[-k:]
                return X[:, top_indices]
            else:
                return X_var
    
    # Normal case - no dimension mismatch
        try:
        # For SelectKBest, we can use transform directly
            if hasattr(self.feature_selector, 'transform'):
                return self.feature_selector.transform(X)
        # For PCA
            elif hasattr(self.pca_model, 'transform') and self.pca_model is not None:
                return self.pca_model.transform(X)
            else:
            # Fallback: manually select features based on the indices
                if hasattr(self.feature_selector, 'get_support'):
                # Get the indices of selected features
                    selected_indices = self.feature_selector.get_support(indices=True)
                # Return only those columns
                    return X[:, selected_indices]
                else:
                    raise ValueError("Cannot determine how to transform features with the current selector")
        except Exception as e:
        # If all else fails, select top k features by variance as a fallback
            logger.error(f"Error applying feature selection: {e}. Falling back to variance-based selection.")
            k = getattr(self.feature_selector, 'k', 20)  # Default to 20 if k is not found
            var = np.var(X, axis=0)
            top_indices = np.argsort(var)[-k:]
            return X[:, top_indices]
    
    def plot_pca_variance(self, n_components: int = None) -> None:
        """
        Plot explained variance ratio of PCA components.
        
        Parameters:
        -----------
        n_components : int
            Number of components to include in the plot (default: all)
        """
        if self.pca_model is None:
            logger.warning("PCA model not fitted yet")
            return
        
        # Get variance ratios
        explained_variance = self.pca_model.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Limit components if specified
        if n_components:
            explained_variance = explained_variance[:n_components]
            cumulative_variance = cumulative_variance[:n_components]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot individual explained variance
        plt.bar(range(1, len(explained_variance) + 1), explained_variance, 
                alpha=0.7, label='Individual Explained Variance')
        
        # Plot cumulative explained variance
        plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, 
                where='mid', label='Cumulative Explained Variance', color='red')
        
        # Add reference line at 95% variance
        plt.axhline(y=0.95, color='gray', linestyle='--', alpha=0.7)
        
        # Determine components for 95% variance explained
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1 if any(cumulative_variance >= 0.95) else len(cumulative_variance)
        plt.axvline(x=n_components_95, color='gray', linestyle='--', alpha=0.7)
        
        # Annotate
        plt.annotate(f'{n_components_95} components for 95% variance', 
                    xy=(n_components_95, 0.95), xytext=(n_components_95 + 1, 0.9),
                    arrowprops=dict(arrowstyle='->'))
        
        plt.xlabel('Number of Components')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Explained Variance')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        logger.info(f"Number of components needed for 95% variance: {n_components_95}")