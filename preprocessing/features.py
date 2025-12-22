"""
Feature Engineering Module for Fuel Efficiency Prediction.

This module handles:
- Feature/target separation
- Data normalization/standardization
- Train-test splitting with reproducibility
- Feature scaling pipeline

CRITICAL INSIGHT ON DATA LEAKAGE:
--------------------------------
Data leakage in regression is ESPECIALLY dangerous because:

1. Test metrics become meaningless - you're evaluating on "seen" distributions
2. Real-world predictions will be worse than you expect
3. Feature importance becomes distorted

Common leakage mistakes:
- Fitting scalers on full dataset, then splitting
- Using test set to tune hyperparameters directly
- Creating features from the entire dataset (e.g., mean encoding)

Always: Fit on train, transform train AND test with the same fitted transformer.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple

from utils.config import RANDOM_SEED, TEST_SPLIT_RATIO, TARGET_COLUMN


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features from target variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset with features and target.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (features, target) tuple.
    """
    df = df.copy()
    
    # Separate target
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])
    
    print(f"[Features] Prepared {X.shape[1]} features and {len(y)} target samples")
    print(f"[Features] Feature columns: {list(X.columns)}")
    
    return X, y


def split_dataset(
    X: pd.DataFrame, 
    y: pd.Series,
    test_size: float = None,
    random_state: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    
    Why reproducibility matters:
    - Consistent comparison across experiments
    - Debugging requires reproducible failures
    - Peer review requires replicable results
    
    Note on stratification:
    - For regression, stratification is less common than classification
    - But for small datasets, binned stratification can help
    - Here we use simple random split given adequate sample size
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    test_size : float, optional
        Fraction for test set. Default from config.
    random_state : int, optional
        Random seed. Default from config.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    if test_size is None:
        test_size = TEST_SPLIT_RATIO
    if random_state is None:
        random_state = RANDOM_SEED
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True  # Shuffle before splitting for random distribution
    )
    
    print(f"\n[Features] Dataset Split (random_state={random_state}):")
    print(f"  Training set: {len(X_train)} samples ({(1-test_size)*100:.0f}%)")
    print(f"  Test set: {len(X_test)} samples ({test_size*100:.0f}%)")
    print(f"  Train MPG range: {y_train.min():.1f} - {y_train.max():.1f}")
    print(f"  Test MPG range: {y_test.min():.1f} - {y_test.max():.1f}")
    
    return X_train, X_test, y_train, y_test


def normalize_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Normalize features using StandardScaler (z-score normalization).
    
    Why standardization matters for neural networks:
    - Features on different scales cause uneven gradient updates
    - Weight initialization assumes roughly unit-variance inputs
    - Convergence is much slower with unnormalized data
    
    Example without normalization:
    - Weight: 2000-5000 lbs
    - Cylinders: 3-8
    - Gradient for weight feature would dominate training
    
    StandardScaler transforms to mean=0, std=1:
    - z = (x - mean) / std
    - All features contribute equally to initial gradients
    
    CRITICAL: We fit the scaler ONLY on training data
    - This prevents leakage of test set statistics
    - Test set is transformed using train set's mean/std
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, StandardScaler]
        Normalized X_train, X_test, and the fitted scaler.
    """
    scaler = StandardScaler()
    
    # Fit ONLY on training data - this is crucial for preventing leakage
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform test data using training statistics
    X_test_scaled = scaler.transform(X_test)
    
    print("\n[Features] Applied StandardScaler normalization:")
    print("  - Fitted on training data only (no leakage)")
    print(f"  - Transformed {X_train_scaled.shape[1]} features")
    
    # Show sample of original vs scaled values
    print("\n  Sample feature scaling (first feature):")
    print(f"    Original mean: {X_train.iloc[:, 0].mean():.2f}")
    print(f"    Original std: {X_train.iloc[:, 0].std():.2f}")
    print(f"    Scaled mean: {X_train_scaled[:, 0].mean():.4f}")
    print(f"    Scaled std: {X_train_scaled[:, 0].std():.4f}")
    
    return X_train_scaled, X_test_scaled, scaler


def prepare_data_pipeline(df: pd.DataFrame) -> dict:
    """
    Run the complete data preparation pipeline.
    
    This is the main entry point that chains all preprocessing steps
    in the correct order to prevent data leakage.
    
    Order is critical:
    1. Separate features from target (safe - no fitting)
    2. Split into train/test (BEFORE any fitting)
    3. Fit scaler on train, transform both (no leakage)
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset from preprocessing.cleaner
        
    Returns
    -------
    dict
        Dictionary containing all prepared data components.
    """
    print("\n" + "="*50)
    print("Starting Feature Preparation Pipeline")
    print("="*50)
    
    # Step 1: Separate features and target
    X, y = prepare_features(df)
    
    # Step 2: Split BEFORE normalization (prevents leakage!)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    
    # Step 3: Normalize using only training statistics
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)
    
    # Convert targets to numpy arrays
    y_train_np = y_train.values
    y_test_np = y_test.values
    
    print("\n" + "-"*50)
    print("[Features] Pipeline complete.")
    print("-"*50 + "\n")
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train_np,
        'y_test': y_test_np,
        'scaler': scaler,
        'feature_names': list(X.columns),
        'X_train_df': X_train,  # Keep original for interpretability
        'X_test_df': X_test
    }


def get_feature_info(feature_names: list, scaler: StandardScaler) -> pd.DataFrame:
    """
    Get information about features including scaling parameters.
    
    Useful for:
    - Understanding what transformations were applied
    - Inverse transforming for interpretation
    - Debugging unexpected model behavior
    
    Parameters
    ----------
    feature_names : list
        List of feature column names.
    scaler : StandardScaler
        Fitted scaler object.
        
    Returns
    -------
    pd.DataFrame
        Feature information table.
    """
    info = pd.DataFrame({
        'feature': feature_names,
        'mean': scaler.mean_,
        'std': np.sqrt(scaler.var_)
    })
    
    print("\n--- Feature Scaling Info ---")
    print(info.to_string(index=False))
    
    return info


if __name__ == "__main__":
    # Quick test when running module directly
    from data.loader import load_data
    from preprocessing.cleaner import clean_data
    
    raw_df = load_data()
    cleaned_df = clean_data(raw_df)
    data = prepare_data_pipeline(cleaned_df)
    
    print(f"\nFinal shapes:")
    print(f"  X_train: {data['X_train'].shape}")
    print(f"  X_test: {data['X_test'].shape}")
    print(f"  y_train: {data['y_train'].shape}")
    print(f"  y_test: {data['y_test'].shape}")
    
    get_feature_info(data['feature_names'], data['scaler'])
