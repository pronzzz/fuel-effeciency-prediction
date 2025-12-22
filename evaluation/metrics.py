"""
Evaluation Metrics Module for Fuel Efficiency Prediction.

This module provides standard regression metrics for model evaluation.

UNDERSTANDING REGRESSION METRICS:
--------------------------------
Different metrics answer different questions:

1. **MAE (Mean Absolute Error)**
   - "On average, how many MPG off is the prediction?"
   - Easy to interpret in real units
   - Treats all errors equally
   - Example: MAE of 2.5 means predictions are ~2.5 MPG off on average

2. **RMSE (Root Mean Squared Error)**
   - "What's the typical error magnitude?"
   - Penalizes large errors more than small ones
   - Sensitive to outliers
   - Same units as target (MPG)
   - Example: RMSE > MAE indicates some large errors exist

3. **MSE (Mean Squared Error)**
   - Square of RMSE
   - Commonly used as loss function (differentiable)
   - Hard to interpret directly (units are MPG²)

4. **R² (Coefficient of Determination)**
   - "How much variance does the model explain?"
   - Range: 0 to 1 (can be negative for bad models)
   - 1.0 = perfect prediction
   - 0.0 = model is no better than predicting mean
"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from typing import Dict


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    MAE = (1/n) * Σ|y_true - y_pred|
    
    Interpretation for Auto MPG:
    - MAE of 2.0 means average prediction is off by 2 MPG
    - Given MPG range of 10-47, this is roughly 5% relative error
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual target values.
    y_pred : np.ndarray
        Predicted target values.
        
    Returns
    -------
    float
        Mean Absolute Error.
    """
    return mean_absolute_error(y_true, y_pred)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    RMSE = sqrt((1/n) * Σ(y_true - y_pred)²)
    
    RMSE vs MAE:
    - RMSE >= MAE always (equality when all errors equal)
    - Large RMSE/MAE ratio indicates presence of large errors
    - Example: MAE=2.0, RMSE=3.0 suggests some predictions are very wrong
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual target values.
    y_pred : np.ndarray
        Predicted target values.
        
    Returns
    -------
    float
        Root Mean Squared Error.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Coefficient of Determination (R²).
    
    R² = 1 - (SS_res / SS_tot)
    where SS_res = Σ(y_true - y_pred)²
    and SS_tot = Σ(y_true - mean(y_true))²
    
    Interpretation:
    - R² = 0.90 means model explains 90% of variance
    - For Auto MPG, R² > 0.85 is typical for good models
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual target values.
    y_pred : np.ndarray
        Predicted target values.
        
    Returns
    -------
    float
        R² score.
    """
    return r2_score(y_true, y_pred)


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all regression metrics at once.
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual target values.
    y_pred : np.ndarray
        Predicted target values.
        
    Returns
    -------
    Dict[str, float]
        Dictionary with all metrics.
    """
    return {
        'mae': calculate_mae(y_true, y_pred),
        'rmse': calculate_rmse(y_true, y_pred),
        'r2': calculate_r2(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred)
    }


def print_evaluation_summary(
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    model_name: str = "Model"
) -> Dict:
    """
    Print a formatted evaluation summary.
    
    Parameters
    ----------
    y_train_true, y_train_pred : Training set actual and predicted
    y_test_true, y_test_pred : Test set actual and predicted
    model_name : str
        Name of the model for display.
        
    Returns
    -------
    Dict
        Dictionary with train and test metrics.
    """
    train_metrics = calculate_all_metrics(y_train_true, y_train_pred)
    test_metrics = calculate_all_metrics(y_test_true, y_test_pred)
    
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Training':<15} {'Test':<15}")
    print(f"{'-'*60}")
    print(f"{'MAE (MPG)':<20} {train_metrics['mae']:<15.3f} {test_metrics['mae']:<15.3f}")
    print(f"{'RMSE (MPG)':<20} {train_metrics['rmse']:<15.3f} {test_metrics['rmse']:<15.3f}")
    print(f"{'R² Score':<20} {train_metrics['r2']:<15.3f} {test_metrics['r2']:<15.3f}")
    print(f"{'='*60}")
    
    # Diagnose overfitting
    if train_metrics['mae'] < test_metrics['mae'] * 0.7:
        print("⚠️  Warning: Significant gap between train and test MAE suggests overfitting.")
    else:
        print("✓ Train/test performance is well-balanced (no severe overfitting).")
    
    print()
    
    return {
        'train': train_metrics,
        'test': test_metrics
    }


def compare_models(model_results: Dict) -> None:
    """
    Compare multiple models' performance.
    
    Parameters
    ----------
    model_results : Dict
        Dictionary of model name -> {'train': metrics, 'test': metrics}
    """
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Model':<25} {'Test MAE':<12} {'Test RMSE':<12} {'Test R²':<12}")
    print("-"*70)
    
    # Sort by test MAE (lower is better)
    sorted_models = sorted(
        model_results.items(),
        key=lambda x: x[1]['test']['mae']
    )
    
    for model_name, metrics in sorted_models:
        test = metrics['test']
        print(f"{model_name:<25} {test['mae']:<12.3f} {test['rmse']:<12.3f} {test['r2']:<12.3f}")
    
    print("="*70)
    
    # Declare winner
    best_model = sorted_models[0][0]
    best_mae = sorted_models[0][1]['test']['mae']
    print(f"\n🏆 Best performing model: {best_model} (Test MAE: {best_mae:.3f} MPG)")
    print()


if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)
    y_true = np.random.uniform(10, 40, 100)
    y_pred = y_true + np.random.normal(0, 2, 100)  # Add noise
    
    metrics = calculate_all_metrics(y_true, y_pred)
    print("Test metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
