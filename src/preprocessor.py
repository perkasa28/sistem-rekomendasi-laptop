"""
Data Preprocessor module for the laptop recommendation system.
"""
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

class LaptopDataPreprocessor:
    """Preprocesses laptop data for the recommendation system."""
    
    def __init__(self):
        """Initialize the preprocessor with necessary scalers and encoders."""
        self.numerical_cols = ['price', 'ram', 'screen_size', 'battery', 'weight']
        self.categorical_cols = ['brand', 'cpu', 'gpu', 'screen_type', 'purpose']
        
        self.scalers: Dict[str, MinMaxScaler] = {}
        self.encoders: Dict[str, OneHotEncoder] = {}
        
    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the preprocessor on the training data.
        
        Args:
            df (pd.DataFrame): Input DataFrame containing laptop data
        """
        # Fit scalers for numerical columns
        for col in self.numerical_cols:
            self.scalers[col] = MinMaxScaler()
            self.scalers[col].fit(df[[col]])
        
        # Fit encoders for categorical columns
        for col in self.categorical_cols:
            self.encoders[col] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.encoders[col].fit(df[[col]])
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform the data into feature vectors.
        
        Args:
            df (pd.DataFrame): Input DataFrame containing laptop data
            
        Returns:
            np.ndarray: Transformed feature vectors
        """
        transformed_features = []
        
        # Transform numerical features
        for col in self.numerical_cols:
            # Ensure numeric type
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Handle any NaN values
            df[col] = df[col].fillna(df[col].mean())
            scaled = self.scalers[col].transform(df[[col]])
            transformed_features.append(scaled)
        
        # Transform categorical features
        for col in self.categorical_cols:
            # Fill any NaN values with most common value
            df[col] = df[col].fillna(df[col].mode()[0])
            encoded = self.encoders[col].transform(df[[col]])
            transformed_features.append(encoded)
        
        # Concatenate all features
        return np.hstack(transformed_features)
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit the preprocessor and transform the data.
        
        Args:
            df (pd.DataFrame): Input DataFrame containing laptop data
            
        Returns:
            np.ndarray: Transformed feature vectors
        """
        self.fit(df)
        return self.transform(df)
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features after transformation.
        
        Returns:
            List[str]: List of feature names
        """
        feature_names = []
        
        # Add numerical column names
        feature_names.extend(self.numerical_cols)
        
        # Add encoded categorical column names
        for col in self.categorical_cols:
            feature_names.extend([
                f"{col}_{val}" for val in self.encoders[col].get_feature_names_out([col])
            ])
        
        return feature_names
    
    def get_feature_dimensions(self) -> Dict[str, Any]:
        """
        Get a mapping of features to their dimensions in the VSM.
        
        Returns:
            Dict: Mapping of feature types to their dimension information
        """
        dimensions = {
            'numerical': self.numerical_cols,
            'categorical': {}
        }
        
        # Add categorical dimensions
        for col in self.categorical_cols:
            dimensions['categorical'][col] = [
                f"{col}_{val}" for val in self.encoders[col].get_feature_names_out([col])
            ]
        
        return dimensions
