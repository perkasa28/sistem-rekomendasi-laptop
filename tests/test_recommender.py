"""
Test module for the laptop recommendation system.
"""
import pytest
import pandas as pd
import numpy as np
from src.data_generator import generate_laptop_dataset
from src.preprocessor import LaptopDataPreprocessor
from src.recommender import LaptopRecommender

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    return generate_laptop_dataset(n_samples=10)

@pytest.fixture
def preprocessor():
    """Create a preprocessor instance."""
    return LaptopDataPreprocessor()

@pytest.fixture
def recommender():
    """Create a recommender instance."""
    return LaptopRecommender()

def test_data_generator():
    """Test data generator functionality."""
    df = generate_laptop_dataset(n_samples=5)
    assert len(df) == 5
    assert all(col in df.columns for col in [
        'brand', 'model', 'price', 'cpu', 'gpu', 'ram',
        'storage', 'screen_size', 'screen_type', 'battery',
        'weight', 'purpose'
    ])

def test_preprocessor_fit_transform(sample_data, preprocessor):
    """Test preprocessor fit_transform functionality."""
    features = preprocessor.fit_transform(sample_data)
    assert isinstance(features, np.ndarray)
    assert len(features) == len(sample_data)

def test_recommender_initialization(recommender):
    """Test recommender initialization."""
    assert recommender.data is None
    assert recommender.features is None
    assert isinstance(recommender.preprocessor, LaptopDataPreprocessor)

def test_recommender_fit(sample_data, recommender):
    """Test recommender fit functionality."""
    recommender.fit(sample_data)
    assert recommender.data is not None
    assert recommender.features is not None
    assert len(recommender.data) == len(sample_data)

def test_recommender_recommendations(sample_data, recommender):
    """Test recommender recommendations functionality."""
    recommender.fit(sample_data)
    
    preferences = {
        'purpose': 'Gaming',
        'price': 15_000_000,
        'weight': 2.0,
        'battery': 4000,
        'screen_size': 15,
        'ram': 16,
        'gpu': 'NVIDIA RTX 3050',
        'cpu': 'Intel i7',
        'screen_type': 'IPS',
        'brand': 'Asus'
    }
    
    recommendations = recommender.recommend(preferences, n_recommendations=3)
    assert len(recommendations) == 3
    assert 'similarity_score' in recommendations.columns

def test_budget_recommendations(sample_data, recommender):
    """Test budget-based recommendations functionality."""
    recommender.fit(sample_data)
    
    preferences = {
        'purpose': 'Student',
        'price': 8_000_000,
        'weight': 1.5,
        'battery': 5000,
        'screen_size': 14,
        'ram': 8,
        'gpu': 'Intel UHD',
        'cpu': 'Intel i5',
        'screen_type': 'IPS',
        'brand': 'Lenovo'
    }
    
    min_budget = 5_000_000
    max_budget = 10_000_000
    
    recommendations = recommender.get_recommendations_by_budget(
        preferences,
        min_budget,
        max_budget,
        n_recommendations=3
    )
    
    assert len(recommendations) <= 3  # Might be less if few laptops in budget range
    assert all(recommendations['price'].between(min_budget, max_budget))
    assert 'similarity_score' in recommendations.columns
