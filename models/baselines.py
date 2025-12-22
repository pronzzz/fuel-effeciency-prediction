"""
Baseline Models for Fuel Efficiency Prediction.

This module provides classical ML baselines to compare against
the neural network. This is CRUCIAL for honest evaluation.

WHY BASELINES MATTER:
--------------------
Most Auto MPG projects show only a neural network and declare success.
This is scientifically incomplete. Baselines answer the key question:

"Does the neural network's complexity provide value over simpler methods?"

For tabular data with ~400 samples:
- Linear Regression often performs surprisingly well
- Random Forest can capture nonlinearities without much tuning
- Neural networks may only provide marginal improvement

If a simple model matches NN performance, prefer the simple model because:
1. Easier to interpret
2. Faster to train and predict
3. Fewer hyperparameters to tune
4. More robust to data drift

FUTURE EXTENSIONS (Comment for later):
- Gradient Boosting (XGBoost, LightGBM)
- Support Vector Regression
- Elastic Net for feature selection
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Tuple, Dict

from utils.config import RANDOM_SEED


def train_linear_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict:
    """
    Train a Linear Regression baseline.
    
    Linear Regression is the simplest meaningful baseline:
    - Assumes linear relationships between features and target
    - MPG = w1*weight + w2*horsepower + ... + bias
    - Closed-form solution (no iterations needed)
    
    For Auto MPG, linear regression is actually quite good because:
    - Weight-MPG relationship is roughly linear (r ≈ -0.83)
    - Feature engineering (one-hot encoding) helps
    - Dataset is well-behaved with no extreme nonlinearities
    
    Parameters
    ----------
    X_train, y_train : Training data
    X_test, y_test : Test data
        
    Returns
    -------
    Dict
        Model and performance metrics.
    """
    print("[Baselines] Training Linear Regression...")
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Metrics
    results = {
        'model': model,
        'name': 'Linear Regression',
        'train_mae': mean_absolute_error(y_train, train_pred),
        'test_mae': mean_absolute_error(y_test, test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
        'predictions': test_pred
    }
    
    print(f"  Train MAE: {results['train_mae']:.3f} MPG")
    print(f"  Test MAE: {results['test_mae']:.3f} MPG")
    
    return results


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = 100
) -> Dict:
    """
    Train a Random Forest baseline.
    
    Random Forest is a strong baseline for tabular data:
    - Captures nonlinear relationships automatically
    - Robust to outliers and noise
    - Built-in feature importance
    - No normalization required (but we already normalized)
    
    For Auto MPG:
    - Can capture curved horsepower-MPG relationship
    - Handles the discrete cylinders feature naturally
    - Usually competitive with neural networks on small datasets
    
    Parameters
    ----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    n_estimators : int
        Number of trees in the forest.
        
    Returns
    -------
    Dict
        Model and performance metrics.
    """
    print(f"[Baselines] Training Random Forest (n_estimators={n_estimators})...")
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=10,  # Prevent overfitting on small dataset
        min_samples_leaf=5,  # Each leaf needs at least 5 samples
        random_state=RANDOM_SEED,
        n_jobs=-1  # Use all CPU cores
    )
    model.fit(X_train, y_train)
    
    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Metrics
    results = {
        'model': model,
        'name': 'Random Forest',
        'train_mae': mean_absolute_error(y_train, train_pred),
        'test_mae': mean_absolute_error(y_test, test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
        'predictions': test_pred,
        'feature_importances': model.feature_importances_
    }
    
    print(f"  Train MAE: {results['train_mae']:.3f} MPG")
    print(f"  Test MAE: {results['test_mae']:.3f} MPG")
    
    return results


def train_ridge_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = 1.0
) -> Dict:
    """
    Train a Ridge Regression (L2 regularized) baseline.
    
    Ridge adds regularization to linear regression:
    - Prevents overfitting when features are correlated
    - Auto MPG has multicollinearity (weight, displacement, horsepower)
    - Ridge shrinks coefficients, reducing variance at cost of bias
    
    Parameters
    ----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    alpha : float
        Regularization strength.
        
    Returns
    -------
    Dict
        Model and performance metrics.
    """
    print(f"[Baselines] Training Ridge Regression (alpha={alpha})...")
    
    model = Ridge(alpha=alpha, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    
    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Metrics
    results = {
        'model': model,
        'name': 'Ridge Regression',
        'train_mae': mean_absolute_error(y_train, train_pred),
        'test_mae': mean_absolute_error(y_test, test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
        'predictions': test_pred
    }
    
    print(f"  Train MAE: {results['train_mae']:.3f} MPG")
    print(f"  Test MAE: {results['test_mae']:.3f} MPG")
    
    return results


def train_all_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Dict]:
    """
    Train all baseline models for comparison.
    
    Parameters
    ----------
    X_train, y_train : Training data
    X_test, y_test : Test data
        
    Returns
    -------
    Dict[str, Dict]
        Dictionary of model results keyed by model name.
    """
    print("\n" + "="*50)
    print("Training Baseline Models")
    print("="*50 + "\n")
    
    baselines = {}
    
    baselines['linear'] = train_linear_regression(X_train, y_train, X_test, y_test)
    baselines['ridge'] = train_ridge_regression(X_train, y_train, X_test, y_test)
    baselines['random_forest'] = train_random_forest(X_train, y_train, X_test, y_test)
    
    print("\n" + "-"*50)
    print("[Baselines] All baseline models trained successfully.")
    print("-"*50 + "\n")
    
    return baselines


def compare_baselines(baselines: Dict[str, Dict]) -> None:
    """
    Print a comparison table of baseline models.
    
    Parameters
    ----------
    baselines : Dict[str, Dict]
        Dictionary of baseline results.
    """
    print("\n" + "="*60)
    print("BASELINE MODEL COMPARISON")
    print("="*60)
    print(f"{'Model':<20} {'Train MAE':<12} {'Test MAE':<12} {'Test RMSE':<12}")
    print("-"*60)
    
    for name, results in baselines.items():
        print(f"{results['name']:<20} "
              f"{results['train_mae']:<12.3f} "
              f"{results['test_mae']:<12.3f} "
              f"{results['test_rmse']:<12.3f}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # Quick test when running module directly
    from data.loader import load_data
    from preprocessing.cleaner import clean_data
    from preprocessing.features import prepare_data_pipeline
    
    raw_df = load_data()
    cleaned_df = clean_data(raw_df)
    data = prepare_data_pipeline(cleaned_df)
    
    baselines = train_all_baselines(
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test']
    )
    
    compare_baselines(baselines)
