"""
Data Loader Module for Fuel Efficiency Prediction System.

This module handles loading the Auto MPG dataset from the UCI Machine Learning
Repository. The dataset contains fuel efficiency data for various automobiles
from the late 1970s to early 1980s.

Key Insight:
-----------
The Auto MPG dataset is a classic regression benchmark, but its real value lies
in understanding the engineering relationships between vehicle attributes and
fuel efficiency - not just achieving low MSE.
"""

import pandas as pd
from utils.config import DATA_URL, COLUMN_NAMES


def load_data(url: str = None) -> pd.DataFrame:
    """
    Load the Auto MPG dataset from UCI ML Repository.
    
    The raw dataset has specific characteristics:
    - No header row (column names provided separately)
    - Missing values marked as '?' (especially in horsepower)
    - Whitespace-delimited format
    - Car name is a multi-word string at the end
    
    Parameters
    ----------
    url : str, optional
        URL to load data from. Defaults to UCI repository.
        
    Returns
    -------
    pd.DataFrame
        Raw dataset with proper column names applied.
        
    Notes
    -----
    This function returns raw data. Use preprocessing.cleaner for
    handling missing values and type conversions.
    """
    if url is None:
        url = DATA_URL
    
    # Read the dataset with proper settings
    # - na_values='?': The dataset uses '?' to indicate missing values
    # - delim_whitespace: Data is separated by variable whitespace
    # - names: Apply column names since dataset has no header
    df = pd.read_csv(
        url,
        names=COLUMN_NAMES,
        na_values='?',
        sep=r'\s+',
        comment='\t',
        skipinitialspace=True
    )
    
    print(f"[Data Loader] Loaded {len(df)} records with {len(df.columns)} columns")
    print(f"[Data Loader] Missing values detected: {df.isnull().sum().sum()}")
    
    return df


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics about the dataset.
    
    This is useful for initial data exploration and understanding
    the distribution of values before preprocessing.
    
    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataset.
        
    Returns
    -------
    dict
        Dictionary containing dataset statistics.
    """
    info = {
        'n_samples': len(df),
        'n_features': len(df.columns) - 1,  # Excluding target
        'missing_values': df.isnull().sum().to_dict(),
        'target_stats': {
            'mean': df['mpg'].mean(),
            'std': df['mpg'].std(),
            'min': df['mpg'].min(),
            'max': df['mpg'].max()
        }
    }
    return info


if __name__ == "__main__":
    # Quick test when running module directly
    df = load_data()
    print("\n--- Dataset Preview ---")
    print(df.head())
    print("\n--- Dataset Info ---")
    print(df.info())
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
