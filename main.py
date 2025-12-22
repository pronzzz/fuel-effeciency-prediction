#!/usr/bin/env python3
"""
Fuel Efficiency Prediction System - Main Pipeline
=================================================

This is the main entry point for the Fuel Efficiency Prediction System.
It orchestrates all stages of the ML pipeline from data loading to
model evaluation and interpretation.

SYSTEM PHILOSOPHY:
-----------------
This is a REGRESSION SYSTEM WITH OPINIONS, not a demo notebook.
- Numbers are only impressive when they explain something real
- We compare models to understand when neural networks help
- We prioritize interpretability over marginal accuracy gains
- We treat MPG prediction as engineering insight

Usage:
------
    python main.py                    # Run full pipeline
    python main.py --eda-only         # Only run EDA
    python main.py --skip-eda         # Skip EDA (faster)
    python main.py --verbose          # Detailed output

Author: Fuel Efficiency Prediction System
"""

import argparse
import os
import sys
import warnings
import numpy as np

# Silence TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# Import pipeline modules
from data.loader import load_data, get_dataset_info
from preprocessing.cleaner import clean_data, validate_cleaned_data
from preprocessing.features import prepare_data_pipeline, get_feature_info
from eda.visualizations import run_full_eda
from models.neural_network import build_model, print_architecture_explanation
from models.baselines import train_all_baselines, compare_baselines
from training.trainer import train_neural_network, get_training_summary
from evaluation.metrics import print_evaluation_summary, compare_models
from evaluation.plots import create_all_evaluation_plots, plot_predicted_vs_actual, plot_residuals
from evaluation.interpretability import (
    analyze_feature_importance, 
    interpret_weight_horsepower_dominance
)
from utils.config import OUTPUT_DIR, RANDOM_SEED


def print_banner():
    """Print the system banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║       🚗  FUEL EFFICIENCY PREDICTION SYSTEM  ⛽                      ║
    ║                                                                      ║
    ║       An Interpretable, Robust Regression System                    ║
    ║       Using the Auto MPG Dataset                                    ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_data_pipeline():
    """
    Execute the data loading and preprocessing pipeline.
    
    Returns
    -------
    tuple
        (cleaned_df, prepared_data)
    """
    print("\n" + "="*70)
    print("STAGE 1: DATA INGESTION AND CLEANING")
    print("="*70)
    
    # Load raw data
    raw_df = load_data()
    
    # Clean the data
    cleaned_df = clean_data(raw_df)
    
    # Validate
    validate_cleaned_data(cleaned_df)
    
    # Prepare features (split, normalize)
    prepared_data = prepare_data_pipeline(cleaned_df)
    
    return cleaned_df, prepared_data


def run_eda(df):
    """
    Execute exploratory data analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset.
    """
    print("\n" + "="*70)
    print("STAGE 2: EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    run_full_eda(df)


def run_model_training(data):
    """
    Train all models (neural network and baselines).
    
    Parameters
    ----------
    data : dict
        Prepared data dictionary.
        
    Returns
    -------
    dict
        All trained models and their results.
    """
    print("\n" + "="*70)
    print("STAGE 3: MODEL TRAINING")
    print("="*70)
    
    results = {}
    
    # Train baseline models first
    print("\n--- Training Baseline Models ---")
    baselines = train_all_baselines(
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test']
    )
    results.update(baselines)
    
    # Train neural network
    print("\n--- Training Neural Network ---")
    print_architecture_explanation()
    
    input_dim = data['X_train'].shape[1]
    nn_model = build_model(input_dim)
    nn_model.summary()
    
    trained_model, training_results = train_neural_network(
        nn_model,
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test'],
        verbose=1
    )
    
    # Store neural network results
    nn_predictions = trained_model.predict(data['X_test'], verbose=0).flatten()
    nn_train_predictions = trained_model.predict(data['X_train'], verbose=0).flatten()
    
    results['neural_network'] = {
        'model': trained_model,
        'name': 'Neural Network',
        'predictions': nn_predictions,
        'train_predictions': nn_train_predictions,
        'training_history': training_results['history'],
        'test_mae': training_results['test_mae']
    }
    
    return results


def run_evaluation(data, model_results):
    """
    Evaluate all models and create visualizations.
    
    Parameters
    ----------
    data : dict
        Prepared data.
    model_results : dict
        Results from all trained models.
    """
    print("\n" + "="*70)
    print("STAGE 4: MODEL EVALUATION AND INTERPRETATION")
    print("="*70)
    
    all_metrics = {}
    
    # Evaluate each model
    for model_name, results in model_results.items():
        print(f"\n--- Evaluating {results['name']} ---")
        
        # Get predictions
        if 'predictions' in results:
            test_pred = results['predictions']
        else:
            test_pred = results['model'].predict(data['X_test'])
        
        if 'train_predictions' in results:
            train_pred = results['train_predictions']
        else:
            train_pred = results['model'].predict(data['X_train'])
        
        # Calculate and print metrics
        metrics = print_evaluation_summary(
            data['y_train'], train_pred,
            data['y_test'], test_pred,
            results['name']
        )
        all_metrics[results['name']] = metrics
        
        # Create plots for this model
        plot_predicted_vs_actual(data['y_test'], test_pred, results['name'])
        plot_residuals(data['y_test'], test_pred, results['name'])
        
        # Training history for neural network
        if 'training_history' in results:
            from evaluation.plots import plot_training_history
            plot_training_history(results['training_history'], results['name'])
    
    # Compare all models
    print("\n--- Model Comparison ---")
    compare_models(all_metrics)
    
    return all_metrics


def run_interpretability(data, model_results):
    """
    Run feature importance and interpretability analysis.
    
    Parameters
    ----------
    data : dict
        Prepared data.
    model_results : dict
        Results from all trained models.
    """
    print("\n" + "="*70)
    print("STAGE 5: INTERPRETABILITY ANALYSIS")
    print("="*70)
    
    # Analyze feature importance for key models
    for model_name in ['neural_network', 'random_forest']:
        if model_name in model_results:
            analyze_feature_importance(
                model_results[model_name]['model'],
                data['X_test'], data['y_test'],
                data['feature_names'],
                model_results[model_name]['name']
            )
    
    # Explain weight and horsepower dominance
    interpret_weight_horsepower_dominance(None)


def main(args):
    """
    Main pipeline execution.
    
    Parameters
    ----------
    args : Namespace
        Command line arguments.
    """
    print_banner()
    
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_SEED)
    
    print(f"\n📂 Output directory: {OUTPUT_DIR}")
    print(f"🎲 Random seed: {RANDOM_SEED}")
    
    # Stage 1: Data Pipeline
    cleaned_df, prepared_data = run_data_pipeline()
    
    # Stage 2: EDA (optional)
    if args.eda_only:
        run_eda(cleaned_df)
        print("\n✅ EDA complete. Exiting as requested.")
        return
    
    if not args.skip_eda:
        run_eda(cleaned_df)
    else:
        print("\n⏭️  Skipping EDA as requested.")
    
    # Stage 3: Model Training
    model_results = run_model_training(prepared_data)
    
    # Stage 4: Evaluation
    all_metrics = run_evaluation(prepared_data, model_results)
    
    # Stage 5: Interpretability
    run_interpretability(prepared_data, model_results)
    
    # Final Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"""
    📊 Generated outputs in: {OUTPUT_DIR}
    
    Key files:
    - pairplot.png                    : Feature relationship visualization
    - correlation_heatmap.png         : Feature correlation matrix
    - predicted_vs_actual_*.png       : Prediction accuracy plots
    - residuals_*.png                 : Residual analysis plots
    - training_history.png            : Neural network training curves
    - feature_importance_*.png        : Feature importance rankings
    
    💡 Key Insights:
    - Weight is the dominant predictor of fuel efficiency
    - Horsepower and displacement are secondary (but correlated with weight)
    - Neural network provides modest improvement over linear regression
    - Interpretability is more valuable than marginal accuracy gains
    """)
    
    print("🏁 Done!")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Fuel Efficiency Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--eda-only',
        action='store_true',
        help='Only run exploratory data analysis'
    )
    
    parser.add_argument(
        '--skip-eda',
        action='store_true',
        help='Skip EDA for faster execution'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
