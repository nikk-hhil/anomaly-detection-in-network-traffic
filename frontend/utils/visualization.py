import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional, Tuple

def create_correlation_heatmap(data: pd.DataFrame, features: List[str], title: str = "Correlation Heatmap") -> go.Figure:
    """
    Create a correlation heatmap for selected features.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    features : List[str]
        List of feature names to include in the heatmap
    title : str
        Title for the heatmap
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    # Calculate correlation matrix
    corr_matrix = data[features].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        aspect="auto",
        title=title
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        width=800
    )
    
    return fig

def create_feature_distribution(data: pd.DataFrame, feature: str, hue: Optional[str] = None) -> go.Figure:
    """
    Create a distribution plot for a feature.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    feature : str
        Name of the feature to plot
    hue : str, optional
        Name of the categorical variable to color by
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    if hue is not None:
        fig = px.histogram(
            data,
            x=feature,
            color=hue,
            marginal="box",
            opacity=0.7,
            title=f"Distribution of {feature} by {hue}"
        )
    else:
        fig = px.histogram(
            data,
            x=feature,
            marginal="box",
            opacity=0.7,
            title=f"Distribution of {feature}"
        )
    
    # Update layout
    fig.update_layout(
        height=500,
        width=800
    )
    
    return fig

def create_scatter_plot(data: pd.DataFrame, x: str, y: str, hue: Optional[str] = None) -> go.Figure:
    """
    Create a scatter plot between two features.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    x : str
        Name of the feature for x-axis
    y : str
        Name of the feature for y-axis
    hue : str, optional
        Name of the categorical variable to color by
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    if hue is not None:
        fig = px.scatter(
            data,
            x=x,
            y=y,
            color=hue,
            opacity=0.7,
            title=f"{x} vs {y} by {hue}"
        )
    else:
        fig = px.scatter(
            data,
            x=x,
            y=y,
            opacity=0.7,
            title=f"{x} vs {y}"
        )
    
    # Update layout
    fig.update_layout(
        height=500,
        width=800
    )
    
    return fig

def create_class_distribution_chart(data: pd.DataFrame, class_column: str) -> go.Figure:
    """
    Create a bar chart showing the distribution of classes.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    class_column : str
        Name of the class column
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    # Get class counts
    class_counts = data[class_column].value_counts().reset_index()
    class_counts.columns = ['Class', 'Count']
    class_counts['Percentage'] = 100 * class_counts['Count'] / class_counts['Count'].sum()
    
    # Create bar chart
    fig = px.bar(
        class_counts,
        x='Class',
        y='Count',
        color='Class',
        text=class_counts['Percentage'].apply(lambda x: f'{x:.1f}%'),
        title=f'Distribution of {class_column}'
    )
    
    # Update layout
    fig.update_layout(
        xaxis={'categoryorder':'total descending'},
        height=500,
        width=800
    )
    
    return fig

def create_confusion_matrix(cm: np.ndarray, classes: np.ndarray, normalize: bool = False) -> go.Figure:
    """
    Create a confusion matrix visualization.
    
    Parameters:
    -----------
    cm : np.ndarray
        Confusion matrix
    classes : np.ndarray
        Class labels
    normalize : bool
        Whether to normalize the confusion matrix
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
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
        text_auto=True,
        title=title
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        width=600,
        coloraxis_showscale=False
    )
    
    return fig

def create_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float) -> go.Figure:
    """
    Create an ROC curve.
    
    Parameters:
    -----------
    fpr : np.ndarray
        False positive rates
    tpr : np.ndarray
        True positive rates
    roc_auc : float
        Area under the ROC curve
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    # Create ROC curve
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
    
    return fig

def create_feature_importance_plot(feature_names: List[str], importance_values: np.ndarray, top_n: int = 20) -> go.Figure:
    """
    Create a bar chart of feature importances.
    
    Parameters:
    -----------
    feature_names : List[str]
        Names of features
    importance_values : np.ndarray
        Importance values for each feature
    top_n : int
        Number of top features to display
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    # Create dataframe for plotting
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Select top features
    top_n = min(top_n, len(feature_names))
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
        height=600,
        width=800
    )
    
    return fig

def create_radar_chart(categories: List[str], values: Dict[str, List[float]]) -> go.Figure:
    """
    Create a radar chart for comparing metrics.
    
    Parameters:
    -----------
    categories : List[str]
        Categories for the radar chart
    values : Dict[str, List[float]]
        Dictionary mapping names to values for each category
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    for name, value_list in values.items():
        fig.add_trace(go.Scatterpolar(
            r=value_list,
            theta=categories,
            fill='toself',
            name=name
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        height=500,
        width=700
    )
    
    return fig

def create_time_series_plot(data: pd.DataFrame, x: str, y: str, hue: Optional[str] = None) -> go.Figure:
    """
    Create a time series plot.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    x : str
        Name of the datetime column
    y : str
        Name of the value column
    hue : str, optional
        Name of the categorical variable to color by
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    if hue is not None:
        fig = px.line(
            data,
            x=x,
            y=y,
            color=hue,
            markers=True,
            title=f"{y} over {x} by {hue}"
        )
    else:
        fig = px.line(
            data,
            x=x,
            y=y,
            markers=True,
            title=f"{y} over {x}"
        )
    
    # Update layout
    fig.update_layout(
        xaxis_title=x,
        yaxis_title=y,
        height=500,
        width=800
    )
    
    return fig

def create_sankey_diagram(source: List[int], target: List[int], value: List[int], 
                         node_labels: List[str], title: str = "Sankey Diagram") -> go.Figure:
    """
    Create a Sankey diagram for flow visualization.
    
    Parameters:
    -----------
    source : List[int]
        Source nodes
    target : List[int]
        Target nodes
    value : List[int]
        Flow values
    node_labels : List[str]
        Labels for nodes
    title : str
        Title for the diagram
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color="blue"
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        )
    )])
    
    fig.update_layout(
        title=title,
        height=600,
        width=800
    )
    
    return fig

def create_3d_scatter(data: pd.DataFrame, x: str, y: str, z: str, hue: Optional[str] = None) -> go.Figure:
    """
    Create a 3D scatter plot.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    x : str
        Name of the x-axis feature
    y : str
        Name of the y-axis feature
    z : str
        Name of the z-axis feature
    hue : str, optional
        Name of the categorical variable to color by
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    if hue is not None:
        fig = px.scatter_3d(
            data,
            x=x,
            y=y,
            z=z,
            color=hue,
            opacity=0.7,
            title=f"3D Scatter Plot: {x} vs {y} vs {z}"
        )
    else:
        fig = px.scatter_3d(
            data,
            x=x,
            y=y,
            z=z,
            opacity=0.7,
            title=f"3D Scatter Plot: {x} vs {y} vs {z}"
        )
    
    # Update layout
    fig.update_layout(
        height=700,
        width=800
    )
    
    return fig

def prepare_dataframe_for_plotting(df):
    """
    Prepare a DataFrame for Plotly by converting non-standard types.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with standard types
    """
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Convert all columns to standard types
    for col in df_copy.columns:
        # Convert pandas extension types to standard types
        if str(df_copy[col].dtype).startswith(('Int', 'Float')):
            df_copy[col] = df_copy[col].astype(float)
        elif str(df_copy[col].dtype) == 'boolean':
            df_copy[col] = df_copy[col].astype(bool)
        
        # Handle object types (strings, etc.)
        elif df_copy[col].dtype == 'object':
            # Try to convert to standard types if possible
            try:
                df_copy[col] = df_copy[col].astype(str)
            except:
                pass
    
    return df_copy