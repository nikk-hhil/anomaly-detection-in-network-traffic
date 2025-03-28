import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataPreprocessor:
    """
    Class for preprocessing network traffic data for anomaly detection.
    """
    
    def __init__(self):
        self.label_encoder = None
        self.feature_encoders = {}
        self.scaler = None
        self.label_mapping = None
    
    def clean_data(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values, duplicates, and invalid entries.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        target_column : str
            Name of the target column
            
        Returns:
        --------
        pd.DataFrame
            Cleaned dataset
        """
        # Create a copy to avoid modifying the original dataframe
        cleaned_data = data.copy()
        
        # 1. Remove duplicate rows
        initial_rows = len(cleaned_data)
        cleaned_data = cleaned_data.drop_duplicates().reset_index(drop=True)
        removed_duplicates = initial_rows - len(cleaned_data)
        print(f"Removed {removed_duplicates} duplicate rows")
        
        # 2. Handle missing values
        missing_percentage = (cleaned_data.isnull().sum() / len(cleaned_data)) * 100
        
        # Display columns with missing values
        columns_with_missing = missing_percentage[missing_percentage > 0]
        if not columns_with_missing.empty:
            print("\nColumns with missing values:")
            for col, pct in columns_with_missing.items():
                print(f"{col}: {pct:.2f}%")
        
        # Drop columns with too many missing values (>50%)
        columns_to_drop = missing_percentage[missing_percentage > 50].index.tolist()
        if columns_to_drop:
            # Ensure we don't drop the target column
            if target_column in columns_to_drop:
                columns_to_drop.remove(target_column)
                print(f"Warning: Target column '{target_column}' had >50% missing values but was kept")
                
            cleaned_data = cleaned_data.drop(columns=columns_to_drop, errors='ignore')
            print(f"Dropped columns with >50% missing values: {columns_to_drop}")
        
        # 3. Handle remaining missing values
        # For numeric columns, fill with median
        numeric_columns = cleaned_data.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_columns:
            if cleaned_data[col].isnull().sum() > 0:
                cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].median())
                print(f"Filled missing values in '{col}' with median")
        
        # For categorical columns, fill with mode
        categorical_columns = cleaned_data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if cleaned_data[col].isnull().sum() > 0:
                cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].mode()[0])
                print(f"Filled missing values in '{col}' with mode")
        
        # 4. Handle infinity values
        cleaned_data = cleaned_data.replace([float('inf'), -float('inf')], np.nan)
        for col in numeric_columns:
            if cleaned_data[col].isna().sum() > 0:
                cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].median())
                print(f"Replaced infinity values in '{col}' with median")
        
        # 5. Check if all missing values are handled
        remaining_missing = cleaned_data.isnull().sum().sum()
        if remaining_missing > 0:
            print(f"WARNING: {remaining_missing} missing values remain")
        else:
            print("All missing values have been handled")
            
        return cleaned_data
    
    def encode_categorical(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Encode categorical features in the dataset.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        target_column : str
            Name of the target column
            
        Returns:
        --------
        pd.DataFrame
            Dataset with encoded categorical features
        """
        # Create a copy to avoid modifying the original dataframe
        encoded_data = data.copy()
        
        # Encode categorical features (except target column)
        categorical_columns = encoded_data.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col != target_column:
                encoder = LabelEncoder()
                encoded_data[col] = encoder.fit_transform(encoded_data[col])
                self.feature_encoders[col] = encoder
                print(f"Encoded categorical column: {col}")
        
        # Check if target column exists and is in categorical_columns
        if target_column in data.columns and target_column not in encoded_data.columns:
            print(f"Warning: Target column '{target_column}' not found in categorical columns")
        
        return encoded_data
    
    def scale_features(self, data: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features and separate features from target.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataset
        target_column : str
            Name of the target column
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Scaled features and encoded target
        """
        # Verify the target column exists
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset. Available columns: {', '.join(data.columns)}")
        
        # Create X (features) and y (target)
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        
        # Print target column information
        print(f"\nTarget column '{target_column}' info:")
        print(f"Data type: {y.dtype}")
        print(f"Unique values: {y.nunique()}")
        print(y.value_counts().head())
        
        # Encode target if it's categorical
        if y.dtype == 'object':
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Create label mapping with safe string representation
            self.label_mapping = {}
            for i, class_label in enumerate(self.label_encoder.classes_):
                try:
                    # Try to safely represent the string
                    safe_label = str(class_label).encode('ascii', 'replace').decode('ascii')
                    self.label_mapping[safe_label] = i
                except:
                    # If that fails, use a placeholder
                    self.label_mapping[f"Class_{i}"] = i
            
            print("\nLabel mapping:")
            for original, encoded in self.label_mapping.items():
                print(f"{original} -> {encoded}")
        else:
            y_encoded = y.values
            print(f"Target column '{target_column}' is already numeric, no encoding needed")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Scaled {X.shape[1]} features")
        
        return X_scaled, y_encoded