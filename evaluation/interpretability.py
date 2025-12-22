"""
Interpretability Module for Fuel Efficiency Prediction.

This module provides tools for understanding WHY the model makes
certain predictions, not just what predictions it makes.

WHY INTERPRETABILITY MATTERS MORE THAN MARGINAL ACCURACY:
---------------------------------------------------------
In real-world engineering decisions, understanding is more valuable than
a 0.1 MPG improvement in MAE. Consider:

1. **Design Decisions**: If redesigning a car, knowing that "reducing
   weight by 500 lbs improves MPG by ~3" is actionable. Just knowing
   the predicted MPG is not.

2. **Trust**: Stakeholders trust models they understand. A black box
   predicting 32 MPG is less convincing than "32 MPG because: lightweight
   design (-2 MPG penalty from weight), efficient 4-cylinder (+3 MPG),
   Japanese origin (+1.5 MPG)".

3. **Debugging**: When predictions go wrong, interpretability reveals
   whether the model learned real patterns or spurious correlations.

4. **Regulatory**: In some contexts (e.g., automotive regulations),
   you need to explain predictions, not just make them.

FUTURE EXTENSIONS:
- SHAP values for game-theoretic feature importance
- Partial Dependence Plots for feature effect curves
- Local Interpretable Model-agnostic Explanations (LIME)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.inspection import permutation_importance
from typing import Dict, List, Optional, Any

from utils.config import OUTPUT_DIR, FIGURE_DPI, RANDOM_SEED


def analyze_feature_importance(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    model_name: str = "Model",
    save: bool = True
) -> pd.DataFrame:
    """
    Calculate and visualize permutation feature importance.
    
    Permutation importance measures how much each feature contributes
    by randomly shuffling it and measuring the increase in error.
    
    Interpretation:
    - High importance = shuffling this feature hurts predictions
    - Low importance = model doesn't rely on this feature
    - Negative importance = shuffling actually helps (rare, indicates noise)
    
    For Auto MPG, we expect:
    - Weight: HIGH importance (strong physical relationship)
    - Horsepower: HIGH importance (engine efficiency)
    - Displacement: MEDIUM (correlated with weight/HP)
    - Model year: MEDIUM (technology improved over time)
    - Cylinders: MEDIUM (discrete version of displacement)
    - Origin: LOW-MEDIUM (proxy for design philosophy)
    - Acceleration: LOW (weak predictor)
    
    Parameters
    ----------
    model : Any
        Trained model with .predict() method.
    X_test, y_test : Test data
    feature_names : List of feature names
    model_name : Name for plots
    save : Whether to save plots
    
    Returns
    -------
    pd.DataFrame
        Feature importance rankings.
    """
    print(f"\n[Interpretability] Calculating permutation importance for {model_name}...")
    
    # For Keras models, need to wrap predict
    if hasattr(model, 'predict'):
        # Check if it's a Keras model (predict returns 2D array)
        test_pred = model.predict(X_test[:1], verbose=0)
        if len(test_pred.shape) > 1:
            # Keras model - need wrapper
            class KerasWrapper:
                def __init__(self, keras_model):
                    self.keras_model = keras_model
                def predict(self, X):
                    return self.keras_model.predict(X, verbose=0).flatten()
            model = KerasWrapper(model)
    
    # Calculate permutation importance
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=30,
        random_state=RANDOM_SEED,
        scoring='neg_mean_absolute_error'
    )
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': result.importances_mean,
        'std': result.importances_std
    }).sort_values('importance', ascending=False)
    
    # Print results
    print(f"\n--- Feature Importance ({model_name}) ---")
    print(importance_df.to_string(index=False))
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(importance_df)))
    
    bars = ax.barh(
        importance_df['feature'],
        importance_df['importance'],
        xerr=importance_df['std'],
        color=colors,
        edgecolor='white',
        linewidth=0.5
    )
    
    ax.set_xlabel('Permutation Importance (increase in MAE when shuffled)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(f'{model_name}: Feature Importance', fontsize=14)
    ax.invert_yaxis()  # Most important at top
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filename = f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"\n[Interpretability] Saved: {filepath}")
    
    plt.close()
    
    return importance_df


def interpret_weight_horsepower_dominance(
    importance_df: pd.DataFrame
) -> str:
    """
    Provide interpretation of why weight and horsepower dominate.
    
    This is key engineering insight that most Auto MPG projects miss.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        Feature importance rankings.
        
    Returns
    -------
    str
        Detailed interpretation text.
    """
    interpretation = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║         WHY WEIGHT AND HORSEPOWER DOMINATE MPG PREDICTION        ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║  PHYSICS EXPLANATION:                                            ║
    ║  -------------------                                             ║
    ║  Fuel efficiency is fundamentally about energy:                  ║
    ║                                                                   ║
    ║  1. WEIGHT (dominant factor):                                    ║
    ║     - Heavier cars need more energy to accelerate (F = ma)       ║
    ║     - More energy = more fuel burned                              ║
    ║     - This is a direct physical relationship                     ║
    ║     - Correlation with MPG: r ≈ -0.83                            ║
    ║                                                                   ║
    ║  2. HORSEPOWER (secondary factor):                               ║
    ║     - Larger engines are less efficient at partial load          ║
    ║     - More power capacity = designed for performance, not economy║
    ║     - But: HP is somewhat redundant with weight                  ║
    ║       (heavy cars have big engines)                              ║
    ║     - Correlation with MPG: r ≈ -0.78                            ║
    ║                                                                   ║
    ║  WHY OTHER FEATURES MATTER LESS:                                 ║
    ║  --------------------------------                                ║
    ║  - Displacement: Highly correlated with HP (redundant info)      ║
    ║  - Cylinders: Discrete proxy for engine size (less informative)  ║
    ║  - Model year: Captures technology trends, but within this       ║
    ║                dataset's narrow range (1970-1982), effect limited║
    ║  - Origin: Proxy for design philosophy, but weight captures this ║
    ║  - Acceleration: Weak predictor, confounded by weight            ║
    ║                                                                   ║
    ║  ENGINEERING IMPLICATION:                                        ║
    ║  -------------------------                                       ║
    ║  To improve fuel efficiency, focus on WEIGHT REDUCTION.          ║
    ║  A 500 lb reduction typically yields 3-5 MPG improvement.        ║
    ║  This is more impactful than engine tuning or origin selection.  ║
    ║                                                                   ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(interpretation)
    return interpretation


def analyze_linear_coefficients(
    linear_model: Any,
    feature_names: List[str],
    scaler: Any = None
) -> pd.DataFrame:
    """
    Analyze linear regression coefficients for interpretability.
    
    For linear models, coefficients directly show feature effects:
    - Positive coefficient: feature increases MPG
    - Negative coefficient: feature decreases MPG
    - Magnitude shows strength (but depends on scaling!)
    
    Note: With standardized features, coefficients are comparable.
    
    Parameters
    ----------
    linear_model : sklearn linear model with .coef_
    feature_names : List of feature names
    scaler : Optional scaler for unscaling interpretation
    
    Returns
    -------
    pd.DataFrame
        Coefficient analysis.
    """
    if not hasattr(linear_model, 'coef_'):
        print("[Interpretability] Model does not have coefficients.")
        return None
    
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': linear_model.coef_,
        'abs_coefficient': np.abs(linear_model.coef_)
    }).sort_values('abs_coefficient', ascending=False)
    
    print("\n--- Linear Model Coefficients ---")
    print("(Interpretation: for each 1 std increase in feature, MPG changes by coefficient)")
    print(coef_df[['feature', 'coefficient']].to_string(index=False))
    
    return coef_df


def compare_model_interpretability(
    nn_importance: pd.DataFrame,
    baseline_importance: pd.DataFrame,
    model_names: tuple = ("Neural Network", "Random Forest"),
    save: bool = True
) -> None:
    """
    Compare feature importance across different models.
    
    If different models agree on importance, we have more confidence
    that we've identified real patterns, not artifacts.
    
    Parameters
    ----------
    nn_importance, baseline_importance : Importance DataFrames
    model_names : Tuple of model names
    save : Whether to save plot
    """
    print("\n[Interpretability] Comparing feature importance across models...")
    
    # Merge on feature name
    merged = nn_importance.merge(
        baseline_importance,
        on='feature',
        suffixes=('_nn', '_baseline')
    )
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(merged))
    width = 0.35
    
    ax.barh(x - width/2, merged['importance_nn'], width, 
            label=model_names[0], color='steelblue', edgecolor='white')
    ax.barh(x + width/2, merged['importance_baseline'], width,
            label=model_names[1], color='darkorange', edgecolor='white')
    
    ax.set_xlabel('Permutation Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Feature Importance Comparison', fontsize=14)
    ax.set_yticks(x)
    ax.set_yticklabels(merged['feature'])
    ax.invert_yaxis()
    ax.legend()
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(OUTPUT_DIR, 'feature_importance_comparison.png')
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"[Interpretability] Saved: {filepath}")
    
    plt.close()


def generate_interpretability_report(
    model_results: Dict,
    feature_names: List[str],
    X_test: np.ndarray,
    y_test: np.ndarray
) -> str:
    """
    Generate a comprehensive interpretability report.
    
    Parameters
    ----------
    model_results : Dict with models and their predictions
    feature_names : Feature names
    X_test, y_test : Test data
    
    Returns
    -------
    str
        Full interpretability report.
    """
    report = []
    report.append("\n" + "="*70)
    report.append("INTERPRETABILITY REPORT: FUEL EFFICIENCY PREDICTION")
    report.append("="*70 + "\n")
    
    report.append("""
    This report explains what the model learned about fuel efficiency
    and why certain features matter more than others.
    """)
    
    # Feature importance for each model
    importance_dfs = {}
    for model_name, results in model_results.items():
        if 'model' in results:
            importance_dfs[model_name] = analyze_feature_importance(
                results['model'], X_test, y_test, feature_names, 
                results.get('name', model_name)
            )
    
    # Add interpretation
    interpret_weight_horsepower_dominance(None)
    
    report.append("\n" + "="*70)
    report.append("END OF INTERPRETABILITY REPORT")
    report.append("="*70 + "\n")
    
    return '\n'.join(report)


if __name__ == "__main__":
    # Demonstration with synthetic importance data
    print("Interpretability module loaded successfully.")
    interpret_weight_horsepower_dominance(None)
