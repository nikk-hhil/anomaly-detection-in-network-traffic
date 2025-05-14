import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import time
from typing import Dict, List, Any
from frontend.utils.visualization import prepare_dataframe_for_plotting

# Add the project root to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from src modules
from src.model_trainer import ModelTrainer
from src.evaluator import ModelEvaluator

def show_training_page():
    """Display the model training and evaluation page."""
    st.title("🤖 Model Training & Evaluation")
    
    # Check if features are available from feature engineering
    if 'features' not in st.session_state or st.session_state.features is None:
        st.warning("Please complete data preprocessing and feature engineering first.")
        st.info("Go to the **Data Upload & Exploration** page to prepare your data.")
        return
    
    # Create tabs for the workflow
    tabs = st.tabs(["Model Selection", "Training", "Evaluation", "Model Comparison"])
    
    # Tab 1: Model Selection
    with tabs[0]:
        show_model_selection_tab()
    
    # Tab 2: Training
    with tabs[1]:
        show_model_training_tab()
    
    # Tab 3: Evaluation
    with tabs[2]:
        show_model_evaluation_tab()
    
    # Tab 4: Model Comparison
    with tabs[3]:
        show_model_comparison_tab()

def show_model_selection_tab():
    """Show model selection options."""
    st.header("Select Models to Train")
    
    # Available models based on your ML service's default models
    available_models = {
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
        "svm": "Support Vector Machine",
        "knn": "K-Nearest Neighbors",
        "mlp": "Neural Network (MLP)",
        "adaboost": "AdaBoost"
    }
    
    # Model selection
    st.subheader("Choose Models")
    
    # Initialize model config in session state if not present
    if 'model_config' not in st.session_state:
        st.session_state.model_config = {
            model_key: {
                # Default to selecting the models your service uses by default
                'selected': model_key in ['logistic_regression', 'decision_tree'],
                'hyperparams': {}
            } for model_key in available_models
        }
    
    # Create columns for model selection
    col1, col2 = st.columns(2)
    
    # First column of models
    first_half = list(available_models.keys())[:len(available_models)//2 + 1]
    with col1:
        for model_key in first_half:
            model_name = available_models[model_key]
            selected = st.checkbox(
                f"Train {model_name}",
                value=st.session_state.model_config[model_key]['selected'],
                key=f"select_{model_key}"
            )
            st.session_state.model_config[model_key]['selected'] = selected
    
    # Second column of models
    second_half = list(available_models.keys())[len(available_models)//2 + 1:]
    with col2:
        for model_key in second_half:
            model_name = available_models[model_key]
            selected = st.checkbox(
                f"Train {model_name}",
                value=st.session_state.model_config[model_key]['selected'],
                key=f"select_{model_key}"
            )
            st.session_state.model_config[model_key]['selected'] = selected
    
    # Additional option for ensemble
    st.markdown("---")
    ensemble_selected = st.checkbox(
        "Create Ensemble (combines selected models)",
        value=st.session_state.model_config.get('ensemble', {}).get('selected', False),
        key="select_ensemble"
    )
    if 'ensemble' not in st.session_state.model_config:
        st.session_state.model_config['ensemble'] = {'selected': ensemble_selected, 'hyperparams': {}}
    else:
        st.session_state.model_config['ensemble']['selected'] = ensemble_selected
    
    # Show which models are selected
    selected_models = [model_key for model_key, config in st.session_state.model_config.items() 
                      if config['selected'] and model_key != 'ensemble']
    
    if selected_models:
        selected_names = [available_models.get(model, model) for model in selected_models]
        st.success(f"Selected models: {', '.join(selected_names)}")
        
        # Hyperparameter tuning note
        st.info("""
        Note: Hyperparameter configuration is handled by the ML service when tuning is enabled.
        Your ModelTrainer will automatically select optimal hyperparameters during training.
        """)
    else:
        st.warning("Please select at least one model to train.")

def show_model_training_tab():
    """Show model training interface."""
    st.header("Train Models")
    
    # Get selected models
    selected_models = [model_key for model_key, config in st.session_state.model_config.items() 
                      if config['selected'] and model_key != 'ensemble']
    
    if not selected_models:
        st.warning("Please select at least one model to train in the Model Selection tab.")
        return
    
    # Initialize model trainer if not already in session state
    if 'model_trainer' not in st.session_state:
        st.session_state.model_trainer = ModelTrainer()
    
    # Show training options
    st.subheader("Training Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider(
            "Test Set Size",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            format="%.2f"
        )
        
        random_state = st.number_input(
            "Random State (for reproducibility)",
            min_value=0,
            max_value=1000,
            value=42,
            step=1
        )
    
    with col2:
        stratify = st.checkbox(
            "Stratified Split",
            value=True,
            help="Maintain class distribution in train and test sets"
        )
        
        tune_hyperparameters = st.checkbox(
            "Tune Hyperparameters",
            value=False,
            help="Use cross-validation to find optimal hyperparameters (slower)"
        )
        
        if tune_hyperparameters:
            cv_folds = st.slider(
                "Cross-Validation Folds",
                min_value=2,
                max_value=10,
                value=3,
                step=1
            )
            
            scoring = st.selectbox(
                "Optimization Metric",
                options=["f1_weighted", "accuracy", "precision_weighted", "recall_weighted"],
                format_func=lambda x: x.replace("_weighted", " (weighted)").replace("_", " ").title()
            )
    
    # Train models button
# Train models button
    if st.button("Train Selected Models", use_container_width=True):
        try:
            with st.spinner("Training models... This may take some time."):
                # Get data
                features_data = st.session_state.features
            
                    # Check for object/string columns in the features
                if features_data is not None:
                    # Make a copy to avoid modifying the original
                    features_data = features_data.copy()
                
                    # Check for and handle object columns (except the target column)
                    object_cols = features_data.select_dtypes(include=['object']).columns.tolist()
                    if st.session_state.target_column in object_cols:
                        object_cols.remove(st.session_state.target_column)
                
                    if object_cols:
                        st.warning(f"Found object/string columns that need encoding: {', '.join(object_cols)}")
                    
                        # Use one-hot encoding for categorical features
                        for col in object_cols:
                            # Create dummy variables
                            dummies = pd.get_dummies(features_data[col], prefix=col, drop_first=False)
                            # Add dummy variables to the dataset
                            features_data = pd.concat([features_data, dummies], axis=1)
                            # Drop the original column
                            features_data = features_data.drop(columns=[col])
                    
                        st.success("Encoded object columns using one-hot encoding.")
            
                # Get features and target
                X = features_data.drop(columns=[st.session_state.target_column]).values

                # Handle target column encoding
                target_col = features_data[st.session_state.target_column]
                if target_col.dtype == 'object':
                    # Use LabelEncoder for the target
                    from sklearn.preprocessing import LabelEncoder
                    label_encoder = LabelEncoder()
                    y = label_encoder.fit_transform(target_col)
                
                    # Store label encoder for future use
                    st.session_state.label_encoder = label_encoder
                    st.session_state.target_classes = label_encoder.classes_
                
                    st.info(f"Encoded target column '{st.session_state.target_column}' using LabelEncoder.")
                else:
                    y = target_col.values
                    # Ensure y is integer type for classification
                    if np.issubdtype(y.dtype, np.number):
                        y = y.astype(int)
            
                feature_names = features_data.drop(columns=[st.session_state.target_column]).columns.tolist()
            
                # Split data
                X_train, X_test, y_train, y_test = st.session_state.model_trainer.split_data(
                    X, y, test_size=test_size, random_state=int(random_state), stratify=stratify
                )
                
                # Store split data in session state for evaluation
                st.session_state.train_test_split = {
                    'X_train': X_train,
                    'X_test': X_test, 
                    'y_train': y_train,
                    'y_test': y_test,
                    'feature_names': feature_names
                }
                
                # Progress placeholder
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                # Reset models in ModelTrainer
                st.session_state.model_trainer.models = {}
                st.session_state.model_trainer.best_model = None
                st.session_state.model_trainer.best_model_name = None
                st.session_state.model_trainer.best_score = 0
                st.session_state.model_trainer.training_time = {}
                
                # Use your ML service's train_models method
                progress_text.text("Training models...")
                
                # Train models using your ML service
                trained_models = st.session_state.model_trainer.train_models(
                    X_train, y_train, 
                    models_to_train=selected_models,
                    tune_hyperparams=tune_hyperparameters,
                    scoring=scoring if tune_hyperparameters else "f1_weighted",
                    cv=cv_folds if tune_hyperparameters else 3,
                    n_jobs=-1
                )
                
                # Train ensemble if selected
                if st.session_state.model_config.get('ensemble', {}).get('selected', False) and len(trained_models) > 1:
                    progress_text.text("Training Ensemble Model...")
                    st.session_state.model_trainer._train_ensemble(X_train, y_train)
                
                progress_bar.progress(1.0)
                progress_text.text("Training completed!")
                
                # Get the trained models from your ML service
                st.session_state.trained_models = st.session_state.model_trainer.models
                
                # Get training times
                st.session_state.training_times = st.session_state.model_trainer.training_time
                
                # Initialize model evaluator
                if 'model_evaluator' not in st.session_state:
                    st.session_state.model_evaluator = ModelEvaluator()
                
                # Evaluate models
                st.session_state.evaluation_results = {}
                
                for model_key, model in st.session_state.trained_models.items():
                    metrics = st.session_state.model_evaluator.evaluate_model(
                        model, X_test, y_test, model_name=model_key
                    )
                    st.session_state.evaluation_results[model_key] = metrics
                
                # Set best model from ModelTrainer
                if st.session_state.model_trainer.best_model is not None:
                    st.session_state.best_model = {
                        'name': st.session_state.model_trainer.best_model_name,
                        'model': st.session_state.model_trainer.best_model
                    }
                else:
                    # If best model not set by trainer, find best by F1 score
                    best_model_key = max(st.session_state.evaluation_results.items(), 
                                       key=lambda x: x[1].get('f1_macro', 0) 
                                           if 'f1_macro' in x[1] 
                                           else x[1].get('f1', 0))[0]
                    
                    st.session_state.best_model = {
                        'name': best_model_key,
                        'model': st.session_state.trained_models[best_model_key]
                    }
                
                # Success message
                st.success(f"Successfully trained {len(st.session_state.trained_models)} models!")
                
                # Summary of models
                st.subheader("Training Results Summary")
                
                # Create summary table
                summary_data = []
                for model_key, model in st.session_state.trained_models.items():
                    if model_key not in st.session_state.evaluation_results:
                        continue
                    
                    metrics = st.session_state.evaluation_results[model_key]
                    f1_score = metrics.get('f1_macro', metrics.get('f1', 0))
                    accuracy = metrics.get('accuracy', 0)
                    training_time = st.session_state.training_times.get(model_key, 0)
                    
                    summary_data.append({
                        'Model': model_key.title().replace('_', ' '),
                        'F1 Score': f1_score,
                        'Accuracy': accuracy,
                        'Training Time (s)': round(training_time, 2)
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # Highlight best model
                if 'best_model' in st.session_state:
                    st.subheader("Best Model")
                    best_model_name = st.session_state.best_model['name'].title().replace('_', ' ')
                    
                    if st.session_state.best_model['name'] in st.session_state.evaluation_results:
                        best_metrics = st.session_state.evaluation_results[st.session_state.best_model['name']]
                        best_f1 = best_metrics.get('f1_macro', best_metrics.get('f1', 0))
                        
                        st.markdown(f"**{best_model_name}** is the best performing model with an F1 score of **{best_f1:.4f}**")
                
                # Prompt to go to evaluation tab
                st.info("Go to the Evaluation tab to see detailed performance metrics.")
        
        except Exception as e:
            st.error(f"Error during model training: {str(e)}")
    
    # Display existing models if available
    if 'trained_models' in st.session_state and st.session_state.trained_models:
        st.subheader("Trained Models")
        
        for model_key, model in st.session_state.trained_models.items():
            model_name = model_key.title().replace('_', ' ')
            training_time = st.session_state.training_times.get(model_key, 0)
            
            # Get evaluation metrics if available
            if 'evaluation_results' in st.session_state and model_key in st.session_state.evaluation_results:
                metrics = st.session_state.evaluation_results[model_key]
                f1_score = metrics.get('f1_macro', metrics.get('f1', 0))
                accuracy = metrics.get('accuracy', 0)
                
                st.markdown(f"**{model_name}**: F1 Score = {f1_score:.4f}, Accuracy = {accuracy:.4f}, Training Time = {training_time:.2f}s")
            else:
                st.markdown(f"**{model_name}**: Training Time = {training_time:.2f}s")

def show_model_evaluation_tab():
    """Show model evaluation results."""
    st.header("Model Evaluation")
    
    # Check if models have been trained
    if 'trained_models' not in st.session_state or not st.session_state.trained_models:
        st.warning("Please train models first in the Training tab.")
        return
    
    # Check if evaluation results are available
    if 'evaluation_results' not in st.session_state or not st.session_state.evaluation_results:
        st.warning("No evaluation results available. Please train models first.")
        return
    
    # Model selection
    model_keys = list(st.session_state.trained_models.keys())
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Select model to evaluate
        selected_model_key = st.selectbox(
            "Select model to evaluate",
            options=model_keys,
            format_func=lambda x: x.title().replace('_', ' '),
            index=model_keys.index(st.session_state.best_model['name']) if 'best_model' in st.session_state else 0
        )
        
        # Get the selected model
        selected_model = st.session_state.trained_models[selected_model_key]
        
        # Display evaluation metrics
        st.subheader("Performance Metrics")
        
        metrics = st.session_state.evaluation_results[selected_model_key]
        
        # Format metrics for display
        accuracy = metrics.get('accuracy', 0)
        
        if 'f1_macro' in metrics:
            # Multiclass case
            precision = metrics.get('precision_macro', 0)
            recall = metrics.get('recall_macro', 0)
            f1 = metrics.get('f1_macro', 0)
        else:
            # Binary case
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            f1 = metrics.get('f1', 0)
        
        # Display metrics
        st.metric("Accuracy", f"{accuracy:.4f}")
        st.metric("Precision", f"{precision:.4f}")
        st.metric("Recall", f"{recall:.4f}")
        st.metric("F1 Score", f"{f1:.4f}")
        
        # Add AUC if available for binary classification
        if 'roc_auc' in metrics:
            auc = metrics.get('roc_auc', 0)
            st.metric("ROC AUC", f"{auc:.4f}")
    
    with col2:
        # Display confusion matrix
        st.subheader("Confusion Matrix")
        
        if 'train_test_split' not in st.session_state:
            st.warning("Test data not found. Please retrain the models.")
            return
        
        X_test = st.session_state.train_test_split['X_test']
        y_test = st.session_state.train_test_split['y_test']
        
        # Calculate confusion matrix
        from sklearn.metrics import confusion_matrix
        
        y_pred = selected_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Get class labels
        unique_classes = np.unique(np.concatenate((y_test, y_pred)))
        
        # Plot interactive confusion matrix
        plot_confusion_matrix(cm, unique_classes, normalize=True)
    
    # Additional evaluation visualizations
    st.markdown("---")
    st.subheader("Additional Visualizations")
    
    # Use tabs for different visualizations
    eval_tabs = st.tabs(["ROC Curve", "Precision-Recall", "Feature Importance", "Error Analysis"])
    
    # Tab 1: ROC Curve (for binary classification)
    with eval_tabs[0]:
        if len(unique_classes) == 2:
            plot_roc_curve(selected_model, X_test, y_test)
        else:
            st.info("ROC Curve is only available for binary classification.")
    
    # Tab 2: Precision-Recall Curve
    with eval_tabs[1]:
        plot_precision_recall_curve(selected_model, X_test, y_test, unique_classes)
    
    # Tab 3: Feature Importance
    with eval_tabs[2]:
        plot_feature_importance(selected_model, st.session_state.train_test_split['feature_names'])
    
    # Tab 4: Error Analysis
    with eval_tabs[3]:
        error_analysis(selected_model, X_test, y_test, unique_classes)

def plot_confusion_matrix(cm: np.ndarray, classes: np.ndarray, normalize: bool = False):
    """Plot interactive confusion matrix."""
    # Normalize confusion matrix if requested
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_text = np.around(cm_norm, decimals=2)
        cm_plot = cm_norm
        title = "Normalized Confusion Matrix"
    else:
        cm_text = cm
        cm_plot = cm
        title = "Confusion Matrix"
    
    # Create heatmap
    fig = px.imshow(
        cm_plot,
        labels=dict(x="Predicted Label", y="True Label", color="Value"),
        x=classes,
        y=classes,
        color_continuous_scale="Blues",
        text_auto=True
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=500,
        width=500,
        coloraxis_showscale=False
    )
    
    # Show plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Display additional metrics
    col1, col2 = st.columns(2)
    
    # Calculate metrics per class
    n_classes = len(classes)
    
    with col1:
        st.subheader("Per-Class Metrics")
        
        # Calculate per-class precision and recall
        precisions = np.zeros(n_classes)
        recalls = np.zeros(n_classes)
        
        for i in range(n_classes):
            # Precision = TP / (TP + FP)
            if cm[:, i].sum() > 0:
                precisions[i] = cm[i, i] / cm[:, i].sum()
            
            # Recall = TP / (TP + FN)
            if cm[i, :].sum() > 0:
                recalls[i] = cm[i, i] / cm[i, :].sum()
        
        # Create dataframe with class metrics
        class_metrics = pd.DataFrame({
            'Class': classes,
            'Precision': precisions,
            'Recall': recalls,
            'F1 Score': 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        })
        
        # Show class metrics
        st.dataframe(class_metrics, use_container_width=True)
    
    with col2:
        st.subheader("Misclassification Heatmap")
        
        # Create off-diagonal (error) matrix
        cm_errors = cm.copy()
        np.fill_diagonal(cm_errors, 0)
        
        # Only show errors if there are any
        if np.sum(cm_errors) > 0:
            fig = px.imshow(
                cm_errors,
                labels=dict(x="Predicted (Incorrect)", y="True", color="Frequency"),
                x=classes,
                y=classes,
                color_continuous_scale="Reds",
                text_auto=True
            )
            
            fig.update_layout(
                title="Misclassification Patterns",
                height=500,
                width=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No misclassifications to display.")

def plot_roc_curve(model, X_test: np.ndarray, y_test: np.ndarray):
    """Plot ROC curve for binary classification."""
    from sklearn.metrics import roc_curve, auc
    
    # Get probability predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Create ROC curve plot
    fig = go.Figure()
    
    # Add ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.4f})',
        line=dict(color='darkblue', width=2)
    ))
    
    # Add diagonal line (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Add thresholds markers
    num_thresholds = 10
    indices = np.linspace(0, len(thresholds) - 1, num_thresholds, dtype=int)
    
    for i in indices:
        threshold = thresholds[i]
        fig.add_trace(go.Scatter(
            x=[fpr[i]], y=[tpr[i]],
            mode='markers',
            marker=dict(size=8),
            name=f'Threshold = {threshold:.2f}',
            hoverinfo='text',
            hovertext=f'Threshold: {threshold:.2f}<br>FPR: {fpr[i]:.4f}<br>TPR: {tpr[i]:.4f}'
        ))
    
    # Update layout
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis=dict(title='False Positive Rate', constrain='domain'),
        yaxis=dict(title='True Positive Rate', scaleanchor="x", scaleratio=1),
        legend=dict(x=0.01, y=0.01, bgcolor='rgba(255, 255, 255, 0.5)'),
        width=700,
        height=500,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanation
    st.markdown("""
    **Interpreting the ROC Curve:**
    - **AUC (Area Under Curve)**: Higher is better (1.0 is perfect, 0.5 is random)
    - **Points on curve**: Each point represents a different threshold value
    - **Top-left corner**: Ideal position (high true positive rate, low false positive rate)
    - **Diagonal line**: Represents random guessing
    """)

def plot_precision_recall_curve(model, X_test: np.ndarray, y_test: np.ndarray, classes: np.ndarray):
    """Plot precision-recall curve."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize
    
    # Check if binary or multiclass
    if len(classes) == 2:
        # Binary classification
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
        avg_precision = average_precision_score(y_test, y_prob)
        
        # Create plot
        fig = go.Figure()
        
        # Add precision-recall curve
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'PR Curve (AP = {avg_precision:.4f})',
            line=dict(color='darkgreen', width=2)
        ))
        
        # Calculate and add F1 score markers
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_threshold_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0
        
        fig.add_trace(go.Scatter(
            x=[recall[best_threshold_idx]],
            y=[precision[best_threshold_idx]],
            mode='markers',
            marker=dict(size=10, color='red'),
            name=f'Best F1 = {f1_scores[best_threshold_idx]:.4f}',
            hoverinfo='text',
            hovertext=f'Threshold: {best_threshold:.2f}<br>F1: {f1_scores[best_threshold_idx]:.4f}<br>Precision: {precision[best_threshold_idx]:.4f}<br>Recall: {recall[best_threshold_idx]:.4f}'
        ))
        
        # Add threshold markers
        num_thresholds = 8
        if len(thresholds) > num_thresholds:
            indices = np.linspace(0, len(thresholds) - 1, num_thresholds, dtype=int)
            
            for i in indices:
                threshold = thresholds[i] if i < len(thresholds) else 0
                fig.add_trace(go.Scatter(
                    x=[recall[i]], y=[precision[i]],
                    mode='markers',
                    marker=dict(size=8),
                    name=f'Threshold = {threshold:.2f}',
                    hoverinfo='text',
                    hovertext=f'Threshold: {threshold:.2f}<br>Precision: {precision[i]:.4f}<br>Recall: {recall[i]:.4f}'
                ))
        
        # Update layout
        fig.update_layout(
            title='Precision-Recall Curve',
            xaxis=dict(title='Recall', range=[0, 1]),
            yaxis=dict(title='Precision', range=[0, 1.05]),
            legend=dict(x=0.01, y=0.01, bgcolor='rgba(255, 255, 255, 0.5)'),
            width=700,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        st.markdown("""
        **Interpreting the Precision-Recall Curve:**
        - **AP (Average Precision)**: Higher is better (1.0 is perfect)
        - **Points on curve**: Each point represents a different threshold value
        - **Top-right corner**: Ideal position (high precision, high recall)
        - **Best F1 Score**: Optimal balance between precision and recall
        """)
    else:
        # Multiclass case - show micro and macro average curves
        y_test_bin = label_binarize(y_test, classes=classes)
        
        # Get probability predictions for each class
        y_prob = model.predict_proba(X_test)
        
        # Calculate precision-recall curves for each class
        precision = {}
        recall = {}
        avg_precision = {}
        
        for i, class_label in enumerate(classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_prob[:, i])
            avg_precision[i] = average_precision_score(y_test_bin[:, i], y_prob[:, i])
        
        # Calculate micro-average precision-recall curve
        precision_micro, recall_micro, _ = precision_recall_curve(
            y_test_bin.ravel(), np.concatenate([y_prob[:, i].reshape(-1, 1) for i in range(len(classes))], axis=1).ravel()
        )
        avg_precision_micro = average_precision_score(
            y_test_bin.ravel(), np.concatenate([y_prob[:, i].reshape(-1, 1) for i in range(len(classes))], axis=1).ravel()
        )
        
        # Create plot
        fig = go.Figure()
        
        # Add curves for each class
        for i, class_label in enumerate(classes):
            fig.add_trace(go.Scatter(
                x=recall[i], y=precision[i],
                mode='lines',
                name=f'Class {class_label} (AP = {avg_precision[i]:.2f})',
                line=dict(width=1.5)
            ))
        
        # Add micro-average curve
        fig.add_trace(go.Scatter(
            x=recall_micro, y=precision_micro,
            mode='lines',
            name=f'Micro-average (AP = {avg_precision_micro:.2f})',
            line=dict(color='gold', width=3)
        ))
        
        # Update layout
        fig.update_layout(
            title='Precision-Recall Curves (One-vs-Rest)',
            xaxis=dict(title='Recall', range=[0, 1]),
            yaxis=dict(title='Precision', range=[0, 1.05]),
            legend=dict(bgcolor='rgba(255, 255, 255, 0.5)'),
            width=700,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        st.markdown("""
        **Interpreting the Multiclass Precision-Recall Curves:**
        - **One curve per class**: Shows precision-recall trade-off for each class vs. all others
        - **Micro-average**: Aggregated curve across all classes
        - **AP (Average Precision)**: Higher is better (1.0 is perfect)
        """)

def plot_feature_importance(model, feature_names: List[str]):
    """Plot feature importance if the model supports it."""
    # Check if model has feature importances
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        
        # Create dataframe for plotting
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        feature_importance = prepare_dataframe_for_plotting(feature_importance)
        
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Select top features
        top_n = min(20, len(feature_names))
        top_features = feature_importance.head(top_n)
        
        # Create horizontal bar chart
        fig = px.bar(
            top_features,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f'Top {top_n} Features by Importance',
            color='Importance',
            color_continuous_scale='viridis'
        )
        
        # Update layout
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance explanation
        st.markdown("""
        **Interpreting Feature Importance:**
        - Higher values indicate features that had more influence on the model's decisions
        - Importance is based on how much each feature contributes to decreasing impurity (for tree-based models)
        - For ensemble models, this represents the average importance across all trees
        """)
    elif hasattr(model, 'coef_'):
        # Linear models have coefficients
        if len(model.classes_) == 2:
            # Binary classification - one set of coefficients
            coefs = model.coef_[0]
            
            # Create dataframe for plotting
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefs
            })
            
            # Sort by absolute coefficient value
            feature_importance['Abs_Coefficient'] = np.abs(feature_importance['Coefficient'])
            feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
            
            # Select top features
            top_n = min(20, len(feature_names))
            top_features = feature_importance.head(top_n)
            
            # Create horizontal bar chart
            fig = px.bar(
                top_features,
                x='Coefficient',
                y='Feature',
                orientation='h',
                title=f'Top {top_n} Features by Coefficient Magnitude',
                color='Coefficient',
                color_continuous_scale='RdBu_r'
            )
            
            # Update layout
            fig.update_layout(
                yaxis=dict(autorange="reversed"),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Coefficient explanation
            st.markdown("""
            **Interpreting Coefficients:**
            - **Positive values** (blue): Feature increases the probability of the positive class
            - **Negative values** (red): Feature decreases the probability of the positive class
            - **Magnitude**: Larger absolute values have stronger effects
            """)
        else:
            # Multiclass - one set of coefficients per class
            st.write("Multiclass model coefficients:")
            
            # Create tabs for each class
            class_tabs = st.tabs([f"Class {c}" for c in model.classes_])
            
            for i, tab in enumerate(class_tabs):
                with tab:
                    coefs = model.coef_[i]
                    
                    # Create dataframe for plotting
                    feature_importance = pd.DataFrame({
                        'Feature': feature_names,
                        'Coefficient': coefs
                    })
                    
                    # Sort by absolute coefficient value
                    feature_importance['Abs_Coefficient'] = np.abs(feature_importance['Coefficient'])
                    feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
                    
                    # Select top features
                    top_n = min(15, len(feature_names))
                    top_features = feature_importance.head(top_n)
                    
                    # Create horizontal bar chart
                    fig = px.bar(
                        top_features,
                        x='Coefficient',
                        y='Feature',
                        orientation='h',
                        title=f'Top {top_n} Features for Class {model.classes_[i]}',
                        color='Coefficient',
                        color_continuous_scale='RdBu_r'
                    )
                    
                    # Update layout
                    fig.update_layout(
                        yaxis=dict(autorange="reversed"),
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("This model type doesn't provide feature importance or coefficients.")

def error_analysis(model, X_test: np.ndarray, y_test: np.ndarray, classes: np.ndarray):
    """Perform error analysis on the model predictions."""
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Identify errors
    errors = y_test != y_pred
    error_indices = np.where(errors)[0]
    
    if len(error_indices) == 0:
        st.success("No errors found in the test set!")
        return
    
    
    # Get error details
    error_details = pd.DataFrame({
        'True Class': y_test[error_indices],
        'Predicted Class': y_pred[error_indices],
        'Error Index': error_indices
    })
    
    error_details = prepare_dataframe_for_plotting(error_details)
    
    # Get probabilities for misclassified examples
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test[error_indices])
        
        # Add prediction confidence
        for i, class_idx in enumerate(classes):
            error_details[f'Prob Class {class_idx}'] = y_prob[:, i]
        
        # Add confidence in prediction
        error_details['Confidence'] = [prob[pred] for prob, pred in zip(y_prob, [np.where(classes == p)[0][0] for p in error_details['Predicted Class']])]
    
    # Group errors by class pairs
    error_pairs = error_details.groupby(['True Class', 'Predicted Class']).size().reset_index(name='Count')
    error_pairs = error_pairs.sort_values('Count', ascending=False)
    error_pairs = prepare_dataframe_for_plotting(error_pairs)
    
    # Display error summary
    st.subheader("Error Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"Total errors: {len(error_indices)} out of {len(y_test)} test samples ({len(error_indices)/len(y_test):.1%})")
        
        # Show most common error types
        st.subheader("Most Common Error Types")
        st.dataframe(error_pairs.head(10), use_container_width=True)
    
    with col2:
        # Create a heatmap of error types
        error_matrix = pd.crosstab(
            error_details['True Class'], 
            error_details['Predicted Class'],
            rownames=['True'], 
            colnames=['Predicted']
        )
        
        fig = px.imshow(
            error_matrix,
            labels=dict(x="Predicted Class", y="True Class", color="Count"),
            title="Error Type Heatmap",
            color_continuous_scale="Reds"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Show confidence distribution for errors
    if 'Confidence' in error_details.columns:
        st.subheader("Prediction Confidence for Errors")
        
        fig = px.histogram(
            error_details,
            x='Confidence',
            nbins=20,
            color='Confidence',
            title="Distribution of Confidence Scores for Errors"
        )
        
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add insights
        low_confidence_errors = error_details[error_details['Confidence'] < 0.5]
        high_confidence_errors = error_details[error_details['Confidence'] >= 0.5]
        
        st.write(f"Low confidence errors (< 0.5): {len(low_confidence_errors)} ({len(low_confidence_errors)/len(error_details):.1%} of errors)")
        st.write(f"High confidence errors (≥ 0.5): {len(high_confidence_errors)} ({len(high_confidence_errors)/len(error_details):.1%} of errors)")
        
        if len(high_confidence_errors) > 0:
            st.warning("""
            High confidence errors indicate potential blind spots in the model.
            These may be challenging edge cases that the model isn't handling well.
            """)

def label_binarize(y, classes):
    """One-hot encode labels."""
    n_samples = len(y)
    n_classes = len(classes)
    y_bin = np.zeros((n_samples, n_classes))
    
    for i, class_val in enumerate(classes):
        y_bin[:, i] = (y == class_val)
    
    return y_bin

def show_model_comparison_tab():
    """Show comparison of multiple trained models."""
    st.header("Model Comparison")
    
    # Check if models have been trained
    if 'trained_models' not in st.session_state or not st.session_state.trained_models:
        st.warning("Please train models first in the Training tab.")
        return
    
    # Check if evaluation results are available
    if 'evaluation_results' not in st.session_state or not st.session_state.evaluation_results:
        st.warning("No evaluation results available. Please train models first.")
        return
    
    # Get available models
    model_keys = list(st.session_state.trained_models.keys())
    model_names = [key.title().replace('_', ' ') for key in model_keys]
    
    # Display comparison options
    st.subheader("Select Models to Compare")
    
    # Model selection
    selected_models = st.multiselect(
        "Select models",
        options=model_keys,
        default=model_keys,
        format_func=lambda x: x.title().replace('_', ' ')
    )
    
    if not selected_models:
        st.info("Please select at least one model to display.")
        return
    
    # Performance metrics comparison
    st.subheader("Performance Metrics Comparison")
    
    # Create comparison dataframe
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1']
    comparison_data = []
    
    for model_key in selected_models:
        metrics = st.session_state.evaluation_results[model_key]
        
        # Get metrics based on classification type
        if 'f1_macro' in metrics:
            # Multiclass case
            model_metrics = {
                'Model': model_key.title().replace('_', ' '),
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision_macro', 0),
                'Recall': metrics.get('recall_macro', 0),
                'F1 Score': metrics.get('f1_macro', 0)
            }
        else:
            # Binary case
            model_metrics = {
                'Model': model_key.title().replace('_', ' '),
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1 Score': metrics.get('f1', 0)
            }
            
            # Add AUC if available
            if 'roc_auc' in metrics:
                model_metrics['ROC AUC'] = metrics.get('roc_auc', 0)
        
        # Add training time
        model_metrics['Training Time (s)'] = st.session_state.training_times.get(model_key, 0)
        
        comparison_data.append(model_metrics)
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(comparison_data)
    
    # Show comparison table
    st.dataframe(comparison_df, use_container_width=True)
    
    comparison_df = prepare_dataframe_for_plotting(comparison_df)
    
    # Create radar chart for model comparison
    st.subheader("Model Performance Radar Chart")
    
    # Prepare data for radar chart
    radar_metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    # Check if all models have ROC AUC
    if all('ROC AUC' in model_data for model_data in comparison_data):
        radar_metrics.append('ROC AUC')
    
    fig = go.Figure()
    
    for model_data in comparison_data:
        model_name = model_data['Model']
        
        fig.add_trace(go.Scatterpolar(
            r=[model_data.get(metric, 0) for metric in radar_metrics],
            theta=radar_metrics,
            fill='toself',
            name=model_name
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add bar chart comparison
    st.subheader("Metrics Comparison Chart")
    
    # Select metric to compare
    metric_to_compare = st.selectbox(
        "Select metric to compare",
        options=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Training Time (s)'] + (['ROC AUC'] if all('ROC AUC' in model_data for model_data in comparison_data) else [])
    )
    
    # Create bar chart
    fig = px.bar(
        comparison_df,
        x='Model',
        y=metric_to_compare,
        color='Model',
        title=f'{metric_to_compare} Comparison'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Model',
        yaxis_title=metric_to_compare,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Save best model button
    st.subheader("Save Best Model")
    
    if st.button("Save Best Model for Prediction"):
        # Set the best model for prediction
        if 'best_model' in st.session_state:
            st.session_state.prediction_model = st.session_state.best_model['model']
            st.session_state.prediction_model_name = st.session_state.best_model['name']
            
            st.success(f"Saved {st.session_state.best_model['name'].title().replace('_', ' ')} as the model for prediction.")
            st.info("Go to the Anomaly Detection page to use this model for predictions.")
        else:
            st.warning("No best model available. Please train models first.")

if __name__ == "__main__":
    # For testing the page in isolation
    import streamlit as st
    
    # Initialize session state
    if 'model_trainer' not in st.session_state:
        st.session_state.model_trainer = ModelTrainer()
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = {}
    if 'features' not in st.session_state:
        # Create dummy features for testing
        st.session_state.features = pd.DataFrame(np.random.randn(100, 5), columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
        st.session_state.features['target'] = np.random.randint(0, 2, size=100)
        st.session_state.target_column = 'target'
    
    # Display the page
    show_training_page()