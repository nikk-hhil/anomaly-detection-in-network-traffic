import pandas as pd
import glob
import os
from typing import List, Optional, Tuple, Dict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_dataset(folder_path: str) -> pd.DataFrame:
    """
    Load and merge CSV files from the specified folder.
    
    Parameters:
    -----------
    folder_path : str
        Path to the folder containing CSV files
        
    Returns:
    --------
    pd.DataFrame
        Merged dataframe from all CSV files
    """
    file_paths = glob.glob(os.path.join(folder_path, '*.csv'))
    
    if not file_paths:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")
    
    print(f"Found {len(file_paths)} CSV files in {folder_path}")
    
    # Load each file and examine columns
    dataframes = []
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path, low_memory=False)
            print(f"File: {os.path.basename(file_path)}, Shape: {df.shape}")
            dataframes.append(df)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not dataframes:
        raise ValueError("No valid dataframes loaded from CSV files")
    
    # Check for column compatibility before merging
    column_sets = [set(df.columns) for df in dataframes]
    common_columns = set.intersection(*column_sets)
    
    if not common_columns:
        # If no common columns, merge all columns (full outer join)
        merged_data = pd.concat(dataframes, ignore_index=True, sort=False)
        print("Warning: No common columns found across files. Performed full outer join.")
    else:
        # Check if there are significant differences in columns
        all_columns = set.union(*column_sets)
        if len(all_columns) > len(common_columns) * 1.5:  # Arbitrary threshold
            print(f"Warning: Files have significantly different columns. " 
                  f"Common: {len(common_columns)}, Total: {len(all_columns)}")
            print("Only common columns will be used for merging.")
            
            # Use only common columns
            filtered_dataframes = [df[list(common_columns)] for df in dataframes]
            merged_data = pd.concat(filtered_dataframes, ignore_index=True)
        else:
            # Standard merge with all columns
            merged_data = pd.concat(dataframes, ignore_index=True)
    
    print(f"Merged dataset shape: {merged_data.shape}")
    
    return merged_data

def get_dataset_info(data: pd.DataFrame) -> str:
    """
    Print basic information about the dataset and identify potential target columns.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset to analyze
        
    Returns:
    --------
    str
        Name of the identified target column
    """
    print("\n=== Dataset Information ===")
    print(f"Number of rows: {data.shape[0]}")
    print(f"Number of columns: {data.shape[1]}")
    print(f"Duplicate rows: {data.duplicated().sum()}")
    print(f"Missing values: {data.isnull().sum().sum()}")
    
    # Memory usage
    memory_usage = data.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"Memory usage: {memory_usage:.2f} MB")
    
    # Data types
    print("\n=== Data Types ===")
    print(data.dtypes.value_counts())
    
    # List all column names
    print("\n=== Column Names ===")
    for i, col in enumerate(data.columns):
        print(f"{i+1}. {col}")
    
    # Try to identify potential target columns
    potential_target_columns = []
    target_keywords = ['label', 'class', 'target', 'attack', 'category']
    
    for col in data.columns:
        # Check if any keywords appear in the column name (case-insensitive)
        if any(keyword in col.lower() for keyword in target_keywords):
            potential_target_columns.append(col)
        
        # Also check if the column has limited unique values (typical for target columns)
        unique_count = data[col].nunique()
        if unique_count > 1 and unique_count <= 15 and data[col].dtype == 'object':
            if col not in potential_target_columns:
                potential_target_columns.append(col)
    
    if potential_target_columns:
        print("\n=== Potential Target Columns ===")
        for i, col in enumerate(potential_target_columns):
            value_counts = data[col].value_counts()
            print(f"{i+1}. {col} (unique values: {len(value_counts)})")
            print(value_counts.head(10))
            print()
        
        # Check for 'Label' with different casing
        for col in data.columns:
            if col.lower() == 'label':
                return col
        
        # If no 'Label' column found, return the first potential target column
        return potential_target_columns[0]
    else:
        print("\nWARNING: Could not identify any potential target columns.")
        return None

def get_column_statistics(data: pd.DataFrame) -> Dict:
    """
    Get detailed statistics for each column in the dataset.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset to analyze
        
    Returns:
    --------
    Dict
        Dictionary with column statistics
    """
    stats = {}
    
    # Analyze each column
    for col in data.columns:
        col_stats = {
            'dtype': str(data[col].dtype),
            'missing': data[col].isnull().sum(),
            'missing_percent': round(data[col].isnull().sum() / len(data) * 100, 2),
            'unique': data[col].nunique()
        }
        
        # Add numeric stats if applicable
        if pd.api.types.is_numeric_dtype(data[col]):
            col_stats.update({
                'min': data[col].min() if not data[col].empty else None,
                'max': data[col].max() if not data[col].empty else None,
                'mean': data[col].mean() if not data[col].empty else None,
                'median': data[col].median() if not data[col].empty else None,
                'std': data[col].std() if not data[col].empty else None,
                'zeros': (data[col] == 0).sum(),
                'zeros_percent': round((data[col] == 0).sum() / len(data) * 100, 2)
            })
        
        stats[col] = col_stats
    
    return stats

def save_dataset(data: pd.DataFrame, output_path: str) -> None:
    """
    Save the processed dataset to a file.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset to save
    output_path : str
        Path where to save the dataset
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the dataset
    data.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")

def split_large_dataset(data: pd.DataFrame, output_dir: str, chunk_size: int = 100000) -> List[str]:
    """
    Split a large dataset into smaller chunks for easier processing.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Large dataset to split
    output_dir : str
        Directory to save the chunks
    chunk_size : int
        Number of rows per chunk
        
    Returns:
    --------
    List[str]
        List of paths to the chunk files
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate number of chunks
    n_chunks = len(data) // chunk_size + (1 if len(data) % chunk_size != 0 else 0)
    
    print(f"Splitting dataset into {n_chunks} chunks of approximately {chunk_size} rows each")
    
    chunk_paths = []
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(data))
        
        chunk = data.iloc[start_idx:end_idx].copy()
        
        chunk_path = os.path.join(output_dir, f"chunk_{i+1}.csv")
        chunk.to_csv(chunk_path, index=False)
        
        chunk_paths.append(chunk_path)
        print(f"Saved chunk {i+1}/{n_chunks} with {len(chunk)} rows to {chunk_path}")
    
    return chunk_paths