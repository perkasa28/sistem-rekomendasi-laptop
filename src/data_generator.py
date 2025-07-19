"""
Data Generator module for creating dummy laptop dataset.
"""
from typing import Dict, List
import numpy as np
import pandas as pd

def generate_laptop_dataset(n_samples: int = 150) -> pd.DataFrame:
    """
    Generate a dummy dataset of laptop specifications.
    
    Args:
        n_samples (int): Number of laptop entries to generate
        
    Returns:
        pd.DataFrame: DataFrame containing generated laptop data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define price ranges based on purpose
    price_ranges = {
        'Student': (5_000_000, 12_000_000),
        'Office': (8_000_000, 15_000_000),
        'Creator': (12_000_000, 20_000_000),
        'Gaming': (15_000_000, 25_000_000)
    }
    np.random.seed(42)  # For reproducibility
    
    # Define possible values for categorical features
    categories: Dict[str, List[str]] = {
        'brand': ['Asus', 'Lenovo', 'HP', 'Dell', 'Acer', 'MSI'],
        'cpu': ['Intel i3', 'Intel i5', 'Intel i7', 'AMD Ryzen 5', 'AMD Ryzen 7'],
        'gpu': ['Intel UHD', 'NVIDIA GTX 1650', 'NVIDIA RTX 3050', 'NVIDIA RTX 3060', 'AMD Radeon'],
        'storage_type': ['SSD', 'HDD'],
        'storage_capacity': ['256GB', '512GB', '1TB'],
        'screen_type': ['IPS', 'OLED', 'TN'],
        'purpose': ['Gaming', 'Office', 'Student', 'Creator']
    }
    
    # Generate purposes first
    purposes = np.random.choice(categories['purpose'], n_samples)
    
    # Generate prices based on purpose
    prices = []
    for purpose in purposes:
        min_price, max_price = price_ranges[purpose]
        prices.append(np.random.randint(min_price, max_price))
    
    # Generate data
    laptops = {
        'brand': np.random.choice(categories['brand'], n_samples),
        'model': [f'Model-{i:03d}' for i in range(n_samples)],
        'price': prices,
        'cpu': np.random.choice(categories['cpu'], n_samples),
        'gpu': np.random.choice(categories['gpu'], n_samples),
        'ram': np.random.choice([4, 8, 16, 32], n_samples),
        'storage': [f'{np.random.choice(categories["storage_capacity"])} {np.random.choice(categories["storage_type"])}' 
                   for _ in range(n_samples)],
        'screen_size': np.random.choice([13, 14, 15, 17], n_samples),
        'screen_type': np.random.choice(categories['screen_type'], n_samples),
        'battery': np.random.randint(3000, 6000, n_samples),
        'weight': np.random.uniform(1.2, 2.8, n_samples).round(2),
        'purpose': np.random.choice(categories['purpose'], n_samples)
    }
    
    return pd.DataFrame(laptops)

if __name__ == "__main__":
    # Generate dataset and save to CSV
    df = generate_laptop_dataset()
    df.to_csv('../data/dummy_laptops.csv', index=False)
    print("Dataset generated successfully!")
