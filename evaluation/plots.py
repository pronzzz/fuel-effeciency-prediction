"""
Evaluation Plots Module for Fuel Efficiency Prediction.

This module creates diagnostic visualizations for model evaluation:
- Predicted vs Actual plots
- Residual analysis
- Training history curves

WHAT THESE PLOTS REVEAL:
-----------------------
1. **Predicted vs Actual**: Points should cluster around diagonal
   - Above line = underprediction
   - Below line = overprediction
   - Spread indicates uncertainty

2. **Residual Plot**: Should show random scatter
   - Pattern = model is missing something
   - Funnel shape = heteroscedasticity
   - Clusters = subpopulation effects

3. **Training History**: Convergence behavior
   - Diverging curves = overfitting
   - Gap at end = should have stopped earlier
   - Both high = underfitting
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Optional

from utils.config import OUTPUT_DIR, FIGURE_DPI, FIGURE_SIZE


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Neural Network",
    save: bool = True
) -> None:
    """
    Create a Predicted vs Actual scatter plot.
    
    This is THE most important regression diagnostic plot:
    - Perfect predictions fall on the y=x diagonal line
    - Vertical spread shows prediction uncertainty
    - Outliers reveal problematic predictions
    
    For Auto MPG, look for:
    - Overall alignment with diagonal (good predictions)
    - Any systematic deviation (bias)
    - Increased spread at extremes (harder to predict very high/low MPG)
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual MPG values.
    y_pred : np.ndarray
        Predicted MPG values.
    model_name : str
        Name for plot title.
    save : bool
        Whether to save the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot of predictions
    ax.scatter(
        y_true, y_pred,
        alpha=0.6,
        edgecolors='white',
        linewidth=0.5,
        s=50,
        c='steelblue'
    )
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot(
        [min_val, max_val], [min_val, max_val],
        'r--', linewidth=2, label='Perfect Prediction'
    )
    
    # Labels and formatting
    ax.set_xlabel('Actual MPG', fontsize=12)
    ax.set_ylabel('Predicted MPG', fontsize=12)
    ax.set_title(f'{model_name}: Predicted vs Actual', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add annotation with metrics
    mae = np.mean(np.abs(y_true - y_pred))
    ax.text(
        0.95, 0.05,
        f'MAE: {mae:.2f} MPG',
        transform=ax.transAxes,
        ha='right',
        fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    
    if save:
        # Clean model name for filename
        filename = f"predicted_vs_actual_{model_name.lower().replace(' ', '_')}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"[Plots] Saved: {filepath}")
    
    plt.close()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Neural Network",
    save: bool = True
) -> None:
    """
    Create residual analysis plots.
    
    Residual = Actual - Predicted
    Positive residual = underprediction
    Negative residual = overprediction
    
    WHAT PATTERNS REVEAL:
    --------------------
    1. **Random scatter**: ✓ Good - model captures main patterns
    2. **Curved pattern**: Model missing nonlinear relationship
    3. **Funnel shape**: Heteroscedasticity - variance changes with prediction
    4. **Clusters**: Subgroups in data (maybe by origin or cylinders)
    
    For Auto MPG:
    - Look for any U-shape at high/low predictions
    - Check if residuals are larger for certain MPG ranges
    
    Parameters
    ----------
    y_true, y_pred : Actual and predicted values
    model_name : str : Name for titles
    save : bool : Whether to save plots
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Residuals vs Predicted
    ax1 = axes[0]
    ax1.scatter(y_pred, residuals, alpha=0.6, edgecolors='white', 
                linewidth=0.5, s=50, c='steelblue')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted MPG', fontsize=12)
    ax1.set_ylabel('Residual (Actual - Predicted)', fontsize=12)
    ax1.set_title(f'{model_name}: Residuals vs Predicted', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Add reference lines at ±2 MAE
    mae = np.mean(np.abs(residuals))
    ax1.axhline(y=2*mae, color='orange', linestyle=':', alpha=0.7, label=f'±2 MAE ({2*mae:.1f})')
    ax1.axhline(y=-2*mae, color='orange', linestyle=':', alpha=0.7)
    ax1.legend()
    
    # Plot 2: Residual Distribution
    ax2 = axes[1]
    sns.histplot(residuals, kde=True, ax=ax2, color='steelblue', edgecolor='white')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Residual (Actual - Predicted)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title(f'{model_name}: Residual Distribution', fontsize=14)
    
    # Add stats
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    ax2.text(
        0.95, 0.95,
        f'Mean: {mean_res:.2f}\nStd: {std_res:.2f}',
        transform=ax2.transAxes,
        ha='right', va='top',
        fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    
    if save:
        filename = f"residuals_{model_name.lower().replace(' ', '_')}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"[Plots] Saved: {filepath}")
    
    plt.close()
    
    # Print interpretation
    print(f"\n[Plots] Residual Analysis for {model_name}:")
    print(f"  Mean Residual: {mean_res:.3f} (should be ~0)")
    print(f"  Std Residual: {std_res:.3f}")
    if abs(mean_res) < 0.5:
        print("  ✓ Model is unbiased (mean residual near zero)")
    else:
        print(f"  ⚠ Model has bias (systematically {'under' if mean_res > 0 else 'over'}predicting)")


def plot_training_history(
    history: Dict,
    model_name: str = "Neural Network",
    save: bool = True
) -> None:
    """
    Plot training and validation loss curves.
    
    These curves reveal training dynamics:
    - Both decreasing = model is learning
    - Training << validation = overfitting
    - Both high = underfitting or need more epochs
    - Early stopping triggered? Look for where val loss stopped improving
    
    Parameters
    ----------
    history : Dict
        Training history with 'loss' and 'val_loss' keys.
    model_name : str
        Name for title.
    save : bool
        Whether to save plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Plot 1: Loss curves
    ax1 = axes[0]
    ax1.plot(epochs, history['loss'], 'b-', linewidth=2, label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title(f'{model_name}: Loss Curves', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Mark best epoch
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = min(history['val_loss'])
    ax1.axvline(x=best_epoch, color='green', linestyle=':', alpha=0.7)
    ax1.scatter([best_epoch], [best_val_loss], color='green', s=100, zorder=5, 
                label=f'Best (Epoch {best_epoch})')
    ax1.legend()
    
    # Plot 2: MAE curves (if available)
    ax2 = axes[1]
    if 'mae' in history:
        ax2.plot(epochs, history['mae'], 'b-', linewidth=2, label='Training MAE')
        ax2.plot(epochs, history['val_mae'], 'r-', linewidth=2, label='Validation MAE')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('MAE (MPG)', fontsize=12)
        ax2.set_title(f'{model_name}: MAE Curves', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        # Show zoomed loss for last portion
        zoom_start = max(0, len(epochs) - 50)
        ax2.plot(epochs[zoom_start:], history['loss'][zoom_start:], 'b-', 
                 linewidth=2, label='Training Loss')
        ax2.plot(epochs[zoom_start:], history['val_loss'][zoom_start:], 'r-', 
                 linewidth=2, label='Validation Loss')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss (MSE)', fontsize=12)
        ax2.set_title(f'{model_name}: Loss Curves (Last 50 Epochs)', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(OUTPUT_DIR, 'training_history.png')
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"[Plots] Saved: {filepath}")
    
    plt.close()
    
    # Print interpretation
    final_train = history['loss'][-1]
    final_val = history['val_loss'][-1]
    gap = final_val - final_train
    
    print(f"\n[Plots] Training History Analysis:")
    print(f"  Total epochs trained: {len(epochs)}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Final train loss: {final_train:.4f}")
    print(f"  Final val loss: {final_val:.4f}")
    print(f"  Train-val gap: {gap:.4f}")
    
    if gap > final_train * 0.5:
        print("  ⚠ Significant overfitting detected (val loss >> train loss)")
    else:
        print("  ✓ Good convergence (train and val losses are close)")


def create_all_evaluation_plots(
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    training_history: Optional[Dict] = None,
    model_name: str = "Neural Network"
) -> None:
    """
    Create all evaluation plots for a model.
    
    Parameters
    ----------
    y_train_true, y_train_pred : Training actual and predicted
    y_test_true, y_test_pred : Test actual and predicted
    training_history : Optional training history dict
    model_name : Name for titles
    """
    print("\n" + "="*50)
    print(f"Creating Evaluation Plots for {model_name}")
    print("="*50 + "\n")
    
    # Test set visualizations
    plot_predicted_vs_actual(y_test_true, y_test_pred, model_name)
    plot_residuals(y_test_true, y_test_pred, model_name)
    
    # Training history (only for neural networks)
    if training_history is not None:
        plot_training_history(training_history, model_name)
    
    print("\n" + "-"*50)
    print("[Plots] All evaluation plots created.")
    print("-"*50 + "\n")


if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)
    y_true = np.random.uniform(10, 40, 80)
    y_pred = y_true + np.random.normal(0, 2, 80)
    
    plot_predicted_vs_actual(y_true, y_pred, "Test Model", save=False)
    plot_residuals(y_true, y_pred, "Test Model", save=False)
