"""
VSM-based Recommendation Engine module for the laptop recommendation system.
"""
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from .preprocessor import LaptopDataPreprocessor

class LaptopRecommender:
    """Recommends laptops based on user preferences using VSM and content-based filtering."""
    
    def __init__(self):
        """Initialize the recommender system."""
        self.preprocessor = LaptopDataPreprocessor()
        self.data: pd.DataFrame = None
        self.vsm_matrix: np.ndarray = None  # VSM representation of laptops
        self.feature_dimensions: Dict = None  # VSM dimension mapping
    
    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the recommender system with laptop data and create VSM representation.
        
        Args:
            df (pd.DataFrame): Input DataFrame containing laptop data
        """
        self.data = df.copy()
        # Transform data into VSM representation
        self.vsm_matrix = self.preprocessor.fit_transform(df)
        # Get VSM dimension mappings
        self.feature_dimensions = self.preprocessor.get_feature_dimensions()
    
    def create_user_vector(self, preferences: Dict[str, Any]) -> np.ndarray:
        """
        Create a VSM query vector from user preferences.
        
        Args:
            preferences (Dict[str, Any): Dictionary containing user preferences
            
        Returns:
            np.ndarray: Normalized user preference vector in VSM space
        """
        # Create base user preferences DataFrame
        user_df = pd.DataFrame([preferences])
        
        # Transform preferences into VSM space using same transformation
        user_vector = self.preprocessor.transform(user_df)
        
        # Apply preference weights to the vector
        weighted_vector = self._apply_preference_weights(user_vector, preferences)
        
        return weighted_vector
    
    def _apply_preference_weights(self, vector: np.ndarray, preferences: Dict[str, Any]) -> np.ndarray:
        """
        Apply user preference weights to VSM vector.
        
        Args:
            vector (np.ndarray): Base VSM vector
            preferences (Dict[str, Any]): User preferences including weights
            
        Returns:
            np.ndarray: Weighted VSM vector
        """
        weighted_vector = vector.copy()
        
        # Apply weights based on user priorities
        if preferences.get('prioritize_performance', False):
            # Increase weight for performance-related dimensions
            for feature in ['cpu', 'gpu', 'ram']:
                if feature in self.feature_dimensions['categorical']:
                    start_idx = self._get_feature_start_idx(feature)
                    end_idx = start_idx + len(self.feature_dimensions['categorical'][feature])
                    weighted_vector[0, start_idx:end_idx] *= 1.5
                    
        if preferences.get('prioritize_mobility', False):
            # Increase weight for mobility-related dimensions
            for feature in ['weight', 'battery']:
                if feature in self.feature_dimensions['numerical']:
                    idx = self.feature_dimensions['numerical'].index(feature)
                    weighted_vector[0, idx] *= 1.5
                    
        if preferences.get('prioritize_display', False):
            # Increase weight for display-related dimensions
            for feature in ['screen_size', 'screen_type']:
                if feature in self.feature_dimensions['numerical']:
                    idx = self.feature_dimensions['numerical'].index(feature)
                    weighted_vector[0, idx] *= 1.5
                elif feature in self.feature_dimensions['categorical']:
                    start_idx = self._get_feature_start_idx(feature)
                    end_idx = start_idx + len(self.feature_dimensions['categorical'][feature])
                    weighted_vector[0, start_idx:end_idx] *= 1.5
        
        # Normalize the weighted vector
        return normalize(weighted_vector)
    
    def get_recommendations_by_budget(self, 
                                    preferences: Dict[str, Any],
                                    min_budget: float,
                                    max_budget: float,
                                    n_recommendations: int = 5) -> pd.DataFrame:
        """
        Get laptop recommendations within a specific budget range using VSM.
        
        Args:
            preferences (Dict[str, Any]): Dictionary containing user preferences
            min_budget (float): Minimum budget
            max_budget (float): Maximum budget
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            pd.DataFrame: Top N recommended laptops within budget
        """
        print(f"Debug - Looking for laptops between {min_budget:,} and {max_budget:,}")
        
        # Filter laptops by budget
        budget_mask = (self.data['price'] >= min_budget) & (self.data['price'] <= max_budget)
        budget_data = self.data[budget_mask].copy()
        
        # Check if any laptops are found in the budget range
        if len(budget_data) == 0:
            # If no laptops in exact range, expand range by 20%
            range_extension = (max_budget - min_budget) * 0.2
            extended_min = min_budget - range_extension
            extended_max = max_budget + range_extension
            budget_mask = (self.data['price'] >= extended_min) & (self.data['price'] <= extended_max)
            budget_data = self.data[budget_mask].copy()
            
            # If still no laptops found, return empty DataFrame with message
            if len(budget_data) == 0:
                empty_df = pd.DataFrame(columns=self.data.columns)
                empty_df['message'] = ["Tidak ada laptop yang ditemukan dalam rentang budget yang dipilih"]
                return empty_df
        
        # Get VSM vectors for filtered laptops
        budget_vectors = self.vsm_matrix[budget_mask]
        
        # Create user preference vector in VSM space
        user_vector = self.create_user_vector(preferences)
        
        # Calculate cosine similarity in VSM space
        similarities = cosine_similarity(user_vector, budget_vectors)[0]
        
        # Get top N recommendations (or all if less than N available)
        n_available = min(n_recommendations, len(budget_data))
        top_indices = np.argsort(similarities)[-n_available:][::-1]
        
        # Return recommended laptops with similarity scores
        recommendations = budget_data.iloc[top_indices].copy()
        recommendations['similarity_score'] = similarities[top_indices]
        
        return recommendations

    def _get_feature_start_idx(self, feature: str) -> int:
        """Get starting index for a categorical feature in the VSM."""
        start_idx = len(self.feature_dimensions['numerical'])
        for cat_feature in sorted(self.feature_dimensions['categorical'].keys()):
            if cat_feature == feature:
                return start_idx
            start_idx += len(self.feature_dimensions['categorical'][cat_feature])
        return start_idx
