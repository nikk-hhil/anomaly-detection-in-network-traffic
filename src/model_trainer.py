import numpy as np
import pandas as pd
import joblib
import time
import os
import logging
import signal
import psutil
import gc
from typing import Dict, List, Tuple, Any, Optional
from contextlib import contextmanager
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import VotingClassifier
from sklearn.base import BaseEstimator
import warnings
import threading
import _thread
import platform

# Get logger
logger = logging.getLogger(__name__)

class TimeoutException(Exception):
    """Exception raised when a function execution times out."""
    pass

@contextmanager
def time_limit(seconds):
    """
    Context manager to limit the execution time of a function.
    Raises TimeoutException if the execution time exceeds the specified limit.
    Works on both Windows and Unix/Linux systems.
    """
    if platform.system() != 'Windows':  # Unix/Linux/Mac
        # Use signal-based approach
        def signal_handler(signum, frame):
            raise TimeoutException(f"Execution timed out after {seconds} seconds")
        
        # Store the previous handler
        previous_handler = signal.getsignal(signal.SIGALRM)
        
        # Set the new handler
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        
        try:
            yield
        finally:
            # Disable the alarm and restore the previous handler
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
    else:  # Windows
        timer = None
        def timeout_handler():
            _thread.interrupt_main()
        
        timer = threading.Timer(seconds, timeout_handler)
        timer.daemon = True
        timer.start()
        
        try:
            yield
        except KeyboardInterrupt:
            # Convert KeyboardInterrupt to TimeoutException
            raise TimeoutException(f"Execution timed out after {seconds} seconds")
        finally:
            if timer:
                timer.cancel()

class MemoryMonitor:
    """Monitor memory usage during model training."""
    
    def __init__(self, threshold_percent=90):
        """
        Initialize memory monitor.
        
        Args:
            threshold_percent (int): Memory usage threshold in percentage.
        """
        self.threshold_percent = threshold_percent
        self.process = psutil.Process(os.getpid())
    
    def check_memory(self):
        """
        Check current memory usage.
        
        Returns:
            tuple: (memory_usage_percent, is_critical)
        """
        memory_info = psutil.virtual_memory()
        memory_usage_percent = memory_info.percent
        is_critical = memory_usage_percent > self.threshold_percent
        
        if is_critical:
            logger.warning(f"Memory usage critical: {memory_usage_percent}%")
            # Force garbage collection to free memory
            gc.collect()
        
        return memory_usage_percent, is_critical

class ModelTrainer:
    """
    Class for training and tuning anomaly detection models with performance optimizations.
    """
    
    def __init__(self, config=None):
        """
        Initialize model trainer with configuration.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration parameters for performance tuning
        """
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        self.training_time = {}
        
        # Performance configuration
        self.config = config or {}
        self.memory_monitor = MemoryMonitor(
            threshold_percent=self.config.get('memory_threshold_percent', 75)  # Lower threshold
        )
        self.training_timeout = self.config.get('training_timeout_seconds', 1800)  # Reduced: 30 min
        self.tuning_timeout = self.config.get('tuning_timeout_seconds', 3600)  # Reduced: 1 hour
        self.sampling_ratio = self.config.get('sampling_ratio', 0.1)  # Default: use only 10% of dataset
        self.fallback_to_randomized = self.config.get('fallback_to_randomized', True)
        self.n_randomized_iterations = self.config.get('n_randomized_iterations', 10)
    
    def _sample_dataset(self, X: np.ndarray, y: np.ndarray, ratio: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Subsample the dataset to reduce computational requirements while preserving
        class distribution and ensuring minimum samples per class.
    
        Parameters:
        -----------
        X : np.ndarray
        Feature matrix
        y : np.ndarray
        Target vector
        ratio : float, optional
        Sampling ratio (if None, use self.sampling_ratio)
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
        Sampled X and y
        """
        if ratio is None:
            ratio = self.sampling_ratio
    
        if ratio < 1.0:
            logger.info(f"Sampling dataset with ratio {ratio}")
        
        # Get unique classes and their counts
            classes, counts = np.unique(y, return_counts=True)
            min_samples_per_class = 5  # Ensure at least 5 samples per class
        
        # For classes with few samples, keep all samples
            rare_class_indices = []
            common_class_indices = []
        
        # Separate indices for rare and common classes
            for cls in classes:
                cls_indices = np.where(y == cls)[0]
                cls_count = len(cls_indices)
            
                if cls_count < min_samples_per_class / ratio:
                    # Keep all samples for very rare classes
                    rare_class_indices.extend(cls_indices)
                else:
                    common_class_indices.extend(cls_indices)
        
        # Convert to numpy arrays
            rare_class_indices = np.array(rare_class_indices)
            common_class_indices = np.array(common_class_indices)
        
        # Sample from common classes
            if len(common_class_indices) > 0:
                # Calculate how many samples to take from common classes
                n_samples_common = int(len(common_class_indices) * ratio)
                sampled_common_indices = np.random.choice(
                    common_class_indices, 
                    size=n_samples_common, 
                    replace=False
                )
            
            # Combine rare class samples with sampled common class samples
                all_sampled_indices = np.concatenate([rare_class_indices, sampled_common_indices])
            
            # Shuffle the indices
                np.random.shuffle(all_sampled_indices)
            
            # Extract the samples
                X_sampled = X[all_sampled_indices]
                y_sampled = y[all_sampled_indices]
            else:
            # If there are no common classes, just use the rare class samples
                X_sampled = X[rare_class_indices]
                y_sampled = y[rare_class_indices]
        
        # Log the class distribution in the sampled dataset
            sampled_classes, sampled_counts = np.unique(y_sampled, return_counts=True)
            logger.info(f"Sampled dataset size: {len(y_sampled)} samples (was {len(y)})")
            logger.info("Class distribution in sampled dataset:")
            for cls, count in zip(sampled_classes, sampled_counts):
                logger.info(f"  Class {cls}: {count} samples")

            return X_sampled, y_sampled
    
    # If no sampling is needed, return the original dataset
        return X, y
    
    def _get_model_params(self, model_name: str, use_reduced_grid: bool = False) -> Dict:
        """
        Get parameter grid for a specific model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        use_reduced_grid : bool
            Whether to use a reduced parameter grid for faster tuning
            
        Returns:
        --------
        Dict
            Dictionary of parameter grid for GridSearchCV
        """
        if use_reduced_grid:
            # Reduced parameter grids for faster tuning
            param_grids = {
                'random_forest': {
                    'n_estimators': [100],
                    'max_depth': [None, 20],
                    'min_samples_split': [2],
                    'min_samples_leaf': [1]
                },
                'gradient_boosting': {
                    'n_estimators': [100],
                    'learning_rate': [0.1],
                    'max_depth': [3]
                },
                'logistic_regression': {
                    'C': [1.0],
                    'solver': ['liblinear'],
                    'max_iter': [200]
                },
                'svm': {
                    'C': [1.0],
                    'kernel': ['rbf'],
                    'gamma': ['scale']
                },
                'decision_tree': {
                    'max_depth': [None, 20],
                    'min_samples_split': [2],
                    'min_samples_leaf': [1]
                },
                'knn': {
                    'n_neighbors': [5],
                    'weights': ['uniform'],
                    'algorithm': ['auto']
                },
                'mlp': {
                    'hidden_layer_sizes': [(100,)],
                    'activation': ['relu'],
                    'solver': ['adam'],
                    'alpha': [0.0001],
                    'max_iter': [200]
                },
                'adaboost': {
                    'n_estimators': [50],
                    'learning_rate': [1.0]
                }
            }
        else:
            # Medium parameter grids for balanced tuning (less extensive than original)
            param_grids = {
                'random_forest': {
                    'n_estimators': [50, 100],
                    'max_depth': [None, 20],
                    'min_samples_split': [2, 5],
                    'class_weight': [None, 'balanced']
                },
                'gradient_boosting': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5],
                    'subsample': [0.8, 1.0]
                },
                'logistic_regression': {
                    'C': [0.1, 1.0, 10.0],
                    'solver': ['liblinear', 'lbfgs'],
                    'max_iter': [200],
                    'class_weight': [None, 'balanced']
                },
                'svm': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto'],
                    'class_weight': [None, 'balanced']
                },
                'decision_tree': {
                    'max_depth': [None, 20],
                    'min_samples_split': [2, 5],
                    'criterion': ['gini', 'entropy'],
                    'class_weight': [None, 'balanced']
                },
                'knn': {
                    'n_neighbors': [3, 5, 9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto']
                },
                'mlp': {
                    'hidden_layer_sizes': [(50,), (100,)],
                    'activation': ['relu'],
                    'solver': ['adam'],
                    'alpha': [0.0001, 0.001],
                    'max_iter': [200]
                },
                'adaboost': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.5, 1.0]
                }
            }
        
        if model_name.lower() in param_grids:
            return param_grids[model_name.lower()]
        else:
            raise ValueError(f"Parameter grid not defined for model: {model_name}")
    
    def _get_minimal_model_params(self, model_name: str) -> Dict:
        """
        Get minimal parameter grid for emergency fallback.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
            
        Returns:
        --------
        Dict
            Dictionary of minimal parameter grid
        """
        # Minimal parameter configurations for emergency fallback
        minimal_params = {
            'random_forest': {
                'n_estimators': 50,
                'max_depth': 10,
                'min_samples_split': 5,
                'n_jobs': self.config.get('n_jobs', -1)
            },
            'gradient_boosting': {
                'n_estimators': 50,
                'learning_rate': 0.1,
                'max_depth': 3
            },
            'logistic_regression': {
                'C': 1.0,
                'solver': 'liblinear',
                'max_iter': 200
            },
            'svm': {
                'C': 1.0,
                'kernel': 'linear',
                'gamma': 'scale'
            },
            'decision_tree': {
                'max_depth': 10,
                'min_samples_split': 5
            },
            'knn': {
                'n_neighbors': 5,
                'weights': 'uniform',
                'algorithm': 'auto',
                'n_jobs': self.config.get('n_jobs', -1)
            },
            'mlp': {
                'hidden_layer_sizes': (50,),
                'activation': 'relu',
                'solver': 'adam',
                'max_iter': 100
            },
            'adaboost': {
                'n_estimators': 50,
                'learning_rate': 1.0
            }
        }
        
        if model_name.lower() in minimal_params:
            return minimal_params[model_name.lower()]
        else:
            return {}
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                  test_size: float = 0.2, random_state: int = 42, 
                  stratify: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        test_size : float
            Proportion of the dataset to include in the test split
        random_state : int
            Random state for reproducibility
        stratify : bool
            Whether to stratify the split based on the target variable
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Train-test split: X_train, X_test, y_train, y_test
        """
        # Use stratified split if requested and possible
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=stratify_param
        )
        
        logger.info(f"\n=== Data Split ===")
        logger.info(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        logger.info(f"Testing set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        
        # Check class distribution
        train_class_dist = np.bincount(y_train) / len(y_train)
        test_class_dist = np.bincount(y_test) / len(y_test)
        
        logger.info("\nClass distribution:")
        for i in range(len(train_class_dist)):
            logger.info(f"Class {i}: {train_class_dist[i]:.2%} (train), {test_class_dist[i]:.2%} (test)")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    models_to_train: List[str] = None, tune_hyperparams: bool = True, 
                    scoring: str = 'f1_weighted', cv: int = 3, 
                    n_jobs: int = -1) -> Dict[str, Any]:
        """
        Train and optionally tune multiple models with performance optimizations.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training target
        models_to_train : List[str]
            List of models to train (options: 'random_forest', 'gradient_boosting', 
            'logistic_regression', 'svm', 'decision_tree', 'knn', 'mlp', 'adaboost', 'ensemble')
        tune_hyperparams : bool
            Whether to tune hyperparameters using GridSearchCV
        scoring : str
            Scoring metric for hyperparameter tuning
        cv : int
            Number of cross-validation folds
        n_jobs : int
            Number of jobs to run in parallel (-1 means using all processors)
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary of trained models
        """
        if models_to_train is None:
        # Start with just two simpler models for initial success
            models_to_train = ['logistic_regression', 'decision_tree']
    
    # Define model constructors
        model_constructors = {
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=500),  # Increased max_iter
            'svm': SVC(probability=True, random_state=42),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'knn': KNeighborsClassifier(),
            'mlp': MLPClassifier(random_state=42),
            'adaboost': AdaBoostClassifier(random_state=42)
        }
    
        logger.info(f"\n=== Training Models ===")
        
        # For imbalanced datasets, adjust the scoring metric
        if len(np.unique(y_train)) > 2:
        # For multiclass, use weighted metrics
            if scoring == 'f1':
                scoring = 'f1_weighted'
            elif scoring == 'precision':
                scoring = 'precision_weighted'
            elif scoring == 'recall':
                scoring = 'recall_weighted'
        
            logger.info(f"Using '{scoring}' scoring for multiclass classification")
        else:
        # For binary classification with imbalanced classes
            class_counts = np.bincount(y_train)
            imbalance_ratio = max(class_counts) / min(class_counts)
        
            if imbalance_ratio > 10:
                # For highly imbalanced datasets, consider using different metrics
                logger.info(f"Dataset is highly imbalanced (ratio: {imbalance_ratio:.2f}). Using '{scoring}' scoring.")
    
    # Check initial memory usage
        memory_usage, is_critical = self.memory_monitor.check_memory()
        logger.info(f"Initial memory usage: {memory_usage:.1f}%")
    
    # Apply dataset sampling if ratio < 1.0
        if self.sampling_ratio < 1.0:
            X_train_sampled, y_train_sampled = self._sample_dataset(X_train, y_train)
        else:
            X_train_sampled, y_train_sampled = X_train, y_train
        
        # If memory is critical before starting, reduce sample size further
        # If memory is critical before starting, reduce sample size further
        if is_critical and self.sampling_ratio > 0.3:
            logger.warning("Memory usage critical before training. Reducing sample size further.")
            X_train_sampled, y_train_sampled = self._sample_dataset(X_train, y_train, ratio=0.3)
    
    # Train selected models with simplified approach for initial success
        for model_name in models_to_train:
            # Skip ensemble initially to get basic models working first
            if model_name.lower() == 'ensemble':
                logger.info("Skipping ensemble model for initial run")
                continue
                
            if model_name not in model_constructors:
                logger.warning(f"Unknown model '{model_name}'. Skipping.")
                continue
        
            logger.info(f"\nTraining {model_name}...")
            start_time = time.time()
        
            base_model = model_constructors[model_name]
        
        # Simplified training approach - try with default parameters first
            try:
                logger.info(f"Training {model_name} with default parameters...")
                model = base_model
            
            # For logistic regression with multiclass, ensure we use appropriate solver
                if model_name == 'logistic_regression' and len(np.unique(y_train_sampled)) > 2:
                    model = LogisticRegression(random_state=42, max_iter=500, 
                                          multi_class='multinomial', 
                                          solver='lbfgs', 
                                          class_weight='balanced')
            
            # Add class weights for decision tree to handle imbalance
                if model_name == 'decision_tree':
                    model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
            
            # Simple fit with timeout - use the cross-platform time_limit
                with time_limit(self.training_timeout):
                    model.fit(X_train_sampled, y_train_sampled)
            
                    self.models[model_name] = model
                    training_time = time.time() - start_time
                    self.training_time[model_name] = training_time
                    logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Evaluate on training data
                    y_pred = model.predict(X_train_sampled)
                    if len(np.unique(y_train_sampled)) > 2:  # multiclass
                        score = f1_score(y_train_sampled, y_pred, average='weighted')
                    else:  # binary
                        score = f1_score(y_train_sampled, y_pred)
            
                    logger.info(f"{model_name} training {scoring}: {score:.4f}")
            
            # Update best model if better
                    if score > self.best_score:
                        self.best_score = score
                        self.best_model = model
                        self.best_model_name = model_name
                
            except Exception as e:
                logger.error(f"Error during training {model_name}: {str(e)}")
            # Try with a more minimal approach
                try:
                    logger.info(f"Trying minimal configuration for {model_name}")
                
                # Create a very simple model with minimal parameters
                    if model_name == 'logistic_regression':
                        model = LogisticRegression(C=1.0, solver='liblinear', max_iter=200)
                    elif model_name == 'decision_tree':
                        model = DecisionTreeClassifier(max_depth=5)
                    else:
                    # For other models, use constructor with minimal params
                        model = model_constructors[model_name]
                
                    # Reduce dataset size further if needed
                    X_minimal, y_minimal = self._sample_dataset(X_train_sampled, y_train_sampled, ratio=0.5)
                
                # Train with timeout
                    with time_limit(self.training_timeout // 2):  # Half the original timeout
                        model.fit(X_minimal, y_minimal)
                
                    self.models[model_name] = model
                    training_time = time.time() - start_time
                    self.training_time[model_name] = training_time
                    logger.info(f"Training with minimal config completed in {training_time:.2f} seconds")
                
                # Evaluate on training data
                    y_pred = model.predict(X_minimal)
                    if len(np.unique(y_minimal)) > 2:  # multiclass
                        score = f1_score(y_minimal, y_pred, average='weighted')
                    else:  # binary
                        score = f1_score(y_minimal, y_pred)
                
                    logger.info(f"{model_name} minimal training {scoring}: {score:.4f}")
                
                # Update best model if better
                    if score > self.best_score:
                        self.best_score = score
                        self.best_model = model
                        self.best_model_name = model_name
                    
                except Exception as e:
                    logger.error(f"Minimal configuration training for {model_name} failed: {str(e)}")
        
        # Force garbage collection after each model
            gc.collect()
    
        if not self.models:
            logger.warning("No models were successfully trained.")
        else:
            logger.info(f"\nBest model: {self.best_model_name} (Score: {self.best_score:.4f})")
    
        return self.models


    def save_models(self, models_dir: str) -> None:
        """
        Save trained models to disk.
    
        Parameters:
        -----------
        models_dir : str
        Directory to save the models
        """
        logger.info(f"\n=== Saving Models ===")
    
    # Create directory if it doesn't exist
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            logger.info(f"Created directory: {models_dir}")
    
    # Check if we have any models to save
        if not self.models:
            logger.warning("No models available to save.")
            return
    
    # Save each model
        for model_name, model in self.models.items():
            try:
                model_path = os.path.join(models_dir, f"{model_name}.joblib")
                joblib.dump(model, model_path)
                logger.info(f"Saved {model_name} to {model_path}")
            except Exception as e:
                logger.error(f"Error saving {model_name}: {e}")
    
    # Save the best model separately
        if self.best_model is not None and self.best_model_name is not None:
            try:
                best_model_path = os.path.join(models_dir, "best_model.joblib")
                joblib.dump(self.best_model, best_model_path)
            
            # Save metadata about the best model
                metadata = {
                    'name': self.best_model_name,
                    'score': self.best_score,
                    'training_time': self.training_time.get(self.best_model_name, 0)
                }
            
                metadata_path = os.path.join(models_dir, "best_model_metadata.joblib")
                joblib.dump(metadata, metadata_path)
            
                logger.info(f"Saved best model ({self.best_model_name}) to {best_model_path}")
            except Exception as e:
                logger.error(f"Error saving best model: {e}")
        else:
            logger.warning("No best model available to save.")
        
        
        
    def _train_with_tuning(self, model_name: str, base_model: BaseEstimator, 
                         X_train: np.ndarray, y_train: np.ndarray, 
                         cv: int, scoring: str, n_jobs: int) -> Optional[BaseEstimator]:
        """
        Train model with hyperparameter tuning, with timeouts and fallbacks.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        base_model : BaseEstimator
            Base model instance
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training target
        cv : int
            Number of cross-validation folds
        scoring : str
            Scoring metric for hyperparameter tuning
        n_jobs : int
            Number of jobs to run in parallel
            
        Returns:
        --------
        Optional[BaseEstimator]
            Trained model or None if all attempts failed
        """
        logger.info(f"Tuning hyperparameters for {model_name} with {cv}-fold cross-validation...")
        
        # Try GridSearchCV with normal parameters first
        try:
            with time_limit(self.tuning_timeout):
                # If dataset is large, use reduced parameter grid
                is_large_dataset = X_train.shape[0] > 10000
                param_grid = self._get_model_params(model_name, use_reduced_grid=is_large_dataset)
                
                # Use StratifiedKFold for imbalanced datasets
                cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
                
                # Check if we should use RandomizedSearchCV instead
                use_randomized = is_large_dataset or self.fallback_to_randomized
                
                if use_randomized:
                    logger.info(f"Using RandomizedSearchCV with {self.n_randomized_iterations} iterations")
                    search = RandomizedSearchCV(
                        base_model, param_grid, scoring=scoring, cv=cv_splitter, 
                        n_jobs=n_jobs, n_iter=self.n_randomized_iterations, random_state=42, verbose=1
                    )
                else:
                    logger.info("Using GridSearchCV")
                    search = GridSearchCV(
                        base_model, param_grid, scoring=scoring, cv=cv_splitter, 
                        n_jobs=n_jobs, verbose=1
                    )
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    search.fit(X_train, y_train)
                
                model = search.best_estimator_
                
                logger.info(f"Best parameters: {search.best_params_}")
                logger.info(f"Best cross-validation score: {search.best_score_:.4f}")
                
                # Save best model
                if search.best_score_ > self.best_score:
                    self.best_score = search.best_score_
                    self.best_model = model
                    self.best_model_name = model_name
                
                return model
                
        except TimeoutException:
            logger.warning(f"Hyperparameter tuning for {model_name} timed out after {self.tuning_timeout} seconds")
            # Fall through to simplified approach
        except MemoryError:
            logger.warning(f"Memory error during hyperparameter tuning for {model_name}")
            # Fall through to simplified approach
        except Exception as e:
            logger.warning(f"Error during hyperparameter tuning for {model_name}: {e}")
            # Fall through to simplified approach
        
        # Try RandomizedSearchCV with reduced parameters
        try:
            logger.info(f"Falling back to RandomizedSearchCV with reduced parameter grid for {model_name}")
            
            # Check memory before continuing
            memory_usage, is_critical = self.memory_monitor.check_memory()
            if is_critical:
                logger.warning("Memory usage critical. Further reducing dataset size.")
                X_train, y_train = self._sample_dataset(X_train, y_train, ratio=0.5)
            
            with time_limit(self.tuning_timeout // 2):  # Half the original timeout
                param_grid = self._get_model_params(model_name, use_reduced_grid=True)
                cv_splitter = StratifiedKFold(n_splits=max(2, cv-1), shuffle=True, random_state=42)
                
                search = RandomizedSearchCV(
                    base_model, param_grid, scoring=scoring, cv=cv_splitter, 
                    n_jobs=n_jobs, n_iter=5, random_state=42, verbose=1
                )
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    search.fit(X_train, y_train)
                
                model = search.best_estimator_
                
                logger.info(f"Best parameters (reduced grid): {search.best_params_}")
                logger.info(f"Best cross-validation score (reduced grid): {search.best_score_:.4f}")
                
                # Save best model
                if search.best_score_ > self.best_score:
                    self.best_score = search.best_score_
                    self.best_model = model
                    self.best_model_name = model_name
                
                return model
                
        except (TimeoutException, MemoryError, Exception) as e:
            logger.warning(f"Reduced parameter tuning for {model_name} failed: {e}")
            # Fall through to minimal approach
        
        # Final fallback: train with default/minimal parameters
        return self._train_with_default_params(model_name, base_model, X_train, y_train)
    
    def _train_with_default_params(self, model_name: str, base_model: BaseEstimator, 
                                X_train: np.ndarray, y_train: np.ndarray) -> Optional[BaseEstimator]:
        """
        Train model with default parameters, with timeouts and fallbacks.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        base_model : BaseEstimator
            Base model instance
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training target
            
        Returns:
        --------
        Optional[BaseEstimator]
            Trained model or None if all attempts failed
        """
        logger.info(f"Training {model_name} with default parameters...")
        
        try:
            with time_limit(self.training_timeout):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = base_model
                    model.fit(X_train, y_train)
                return model
                
        except TimeoutException:
            logger.warning(f"Default parameter training for {model_name} timed out after {self.training_timeout} seconds")
        except MemoryError:
            logger.warning(f"Memory error during default parameter training for {model_name}")
        except Exception as e:
            logger.warning(f"Error during default parameter training for {model_name}: {e}")
        
        # Try with minimal parameters and reduced dataset
        try:
            logger.info(f"Trying minimal configuration for {model_name}")
            
            # Check memory and reduce dataset if needed
            memory_usage, is_critical = self.memory_monitor.check_memory()
            reduced_X, reduced_y = self._sample_dataset(X_train, y_train, ratio=0.3)
            
            with time_limit(self.training_timeout // 2):  # Half the original timeout
                # Create model with minimal parameters
                minimal_params = self._get_minimal_model_params(model_name)
                if model_name == 'random_forest':
                    model = RandomForestClassifier(**minimal_params, random_state=42)
                elif model_name == 'gradient_boosting':
                    model = GradientBoostingClassifier(**minimal_params, random_state=42)
                elif model_name == 'logistic_regression':
                    model = LogisticRegression(**minimal_params, random_state=42)
                elif model_name == 'svm':
                    model = SVC(**minimal_params, probability=True, random_state=42)
                elif model_name == 'decision_tree':
                    model = DecisionTreeClassifier(**minimal_params, random_state=42)
                elif model_name == 'knn':
                    model = KNeighborsClassifier(**minimal_params)
                elif model_name == 'mlp':
                    model = MLPClassifier(**minimal_params, random_state=42)
                elif model_name == 'adaboost':
                    model = AdaBoostClassifier(**minimal_params, random_state=42)
                else:
                    model = base_model
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(reduced_X, reduced_y)
                
                logger.info(f"Successfully trained {model_name} with minimal configuration")
                return model
                
        except (TimeoutException, MemoryError, Exception) as e:
            logger.error(f"Minimal configuration training for {model_name} failed: {e}")
            return None

def _train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray, 
                   cv: int = 3, scoring: str = 'f1_weighted', n_jobs: int = -1) -> None:
    """
    Train an ensemble model using the best of other models.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training target
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric for model evaluation
    n_jobs : int
        Number of jobs to run in parallel
    """
    logger.info("\nTraining Ensemble Model...")
    
    # Need at least 2 models to create an ensemble
    if len(self.models) < 2:
        logger.warning("Not enough trained models to create an ensemble. Train at least 2 models first.")
        return
    
    # Check memory before ensemble training
    memory_usage, is_critical = self.memory_monitor.check_memory()
    if is_critical:
        logger.warning("Memory usage critical before ensemble training. Using reduced dataset.")
        X_train, y_train = self._sample_dataset(X_train, y_train, ratio=0.3)
    
    # Select top 3 models if we have more
    if len(self.models) > 3:
        # Evaluate each model using cross-validation
        from sklearn.model_selection import cross_val_score
        
        model_scores = {}
        for model_name, model in self.models.items():
            try:
                with time_limit(120):  # Timeout for individual model evaluation
                    scores = cross_val_score(model, X_train, y_train, cv=min(cv, 3), 
                                          scoring=scoring, n_jobs=n_jobs)
                    model_scores[model_name] = np.mean(scores)
                    logger.info(f"{model_name} cross-validation {scoring}: {model_scores[model_name]:.4f}")
            except Exception as e:
                logger.error(f"Error evaluating {model_name} for ensemble: {e}")
        
        # Select top models
        top_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        top_model_names = [name for name, _ in top_models]
        logger.info(f"Selected top models for ensemble: {', '.join(top_model_names)}")
        
        # Create ensemble from top models
        ensemble_models = [(name, self.models[name]) for name in top_model_names 
                          if name in self.models]
    else:
        # Use all available models
        ensemble_models = [(name, model) for name, model in self.models.items()]
    
    if not ensemble_models:
        logger.error("No viable models available for ensemble training")
        return
    
    # Create voting classifier
    voting_classifier = VotingClassifier(
        estimators=ensemble_models,
        voting='soft',  # Use probabilities for voting
        n_jobs=n_jobs
    )
    
    # Train the ensemble with timeout
    start_time = time.time()
    try:
        with time_limit(min(600, self.training_timeout // 2)):  # Shorter timeout for ensemble
            voting_classifier.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        self.models['ensemble'] = voting_classifier
        self.training_time['ensemble'] = training_time
        
        logger.info(f"Ensemble model trained in {training_time:.2f} seconds")
        
        # Evaluate ensemble on training data
        y_pred = voting_classifier.predict(X_train)
        if len(np.unique(y_train)) > 2:  # multiclass
            score = f1_score(y_train, y_pred, average='weighted')
        else:  # binary
            score = f1_score(y_train, y_pred)
        
        logger.info(f"Ensemble training {scoring}: {score:.4f}")
        
        # Update best model if ensemble is better
        if score > self.best_score:
            self.best_score = score
            self.best_model = voting_classifier
            self.best_model_name = 'ensemble'
            logger.info("Ensemble is now the best model")
    
    except TimeoutException:
        logger.error("Ensemble training timed out")
    except Exception as e:
        logger.error(f"Error training ensemble: {e}")
        
        # Need at least 2 models to create an ensemble
        if len(self.models) < 2:
            logger.warning("Not enough trained models to create an ensemble. Train at least 2 models first.")
            return