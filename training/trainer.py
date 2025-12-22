"""
Training Module for Fuel Efficiency Prediction.

This module handles the training loop for the neural network,
including callbacks, history tracking, and early stopping.

TRAINING PHILOSOPHY:
-------------------
Training a neural network is NOT just about minimizing loss.
It's about finding a model that generalizes to unseen data.

Key principles:
1. Track both training AND validation loss
2. Stop early if validation loss stops improving
3. Restore the best weights, not the final weights
4. Use a subset of training data for validation during training
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, Dict, Optional

from utils.config import (
    EPOCHS,
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    RANDOM_SEED
)


def create_callbacks() -> list:
    """
    Create training callbacks for the neural network.
    
    Callbacks intercept training at various points to:
    - Monitor progress
    - Implement early stopping
    - Save best model
    
    Early Stopping explained:
    - Monitors validation loss (not training loss!)
    - Waits 'patience' epochs for improvement
    - If no improvement > min_delta, stops training
    - Restores weights from best epoch
    
    Why patience=20?
    - Too low: might stop during temporary plateau
    - Too high: wastes time training overfit epochs
    - 20 epochs is a reasonable balance
    
    Returns
    -------
    list
        List of Keras callbacks.
    """
    callbacks = [
        # Early stopping to prevent overfitting
        keras.callbacks.EarlyStopping(
            monitor='val_loss',           # Watch validation loss
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=EARLY_STOPPING_MIN_DELTA,
            restore_best_weights=True,    # CRITICAL: use best, not last
            verbose=1
        ),
        
        # Reduce learning rate when stuck
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,                   # Halve the learning rate
            patience=10,                  # Wait 10 epochs before reducing
            min_lr=1e-6,                  # Don't go below this
            verbose=1
        )
    ]
    
    return callbacks


def train_neural_network(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = None,
    batch_size: int = None,
    validation_split: float = 0.2,
    verbose: int = 1
) -> Tuple[keras.Model, Dict]:
    """
    Train the neural network with proper validation.
    
    Training strategy:
    1. Use a portion of training data for validation during training
    2. Monitor validation loss to detect overfitting
    3. Stop early if validation loss plateaus
    4. Evaluate final model on held-out test set
    
    Why validation_split on training data?
    - Test set should NEVER be seen during training decisions
    - Validation set guides early stopping and hyperparameter choices
    - Final test set is for unbiased performance estimation only
    
    Parameters
    ----------
    model : keras.Model
        Compiled Keras model.
    X_train, y_train : Training data
    X_test, y_test : Test data (only for final evaluation)
    epochs : int, optional
        Maximum epochs. Default from config.
    batch_size : int, optional
        Batch size. Default from config.
    validation_split : float
        Fraction of training data for validation.
    verbose : int
        Verbosity level (0=silent, 1=progress, 2=one line per epoch)
        
    Returns
    -------
    Tuple[keras.Model, Dict]
        Trained model and training history.
    """
    if epochs is None:
        epochs = EPOCHS
    if batch_size is None:
        batch_size = BATCH_SIZE
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    print("\n" + "="*50)
    print("Starting Neural Network Training")
    print("="*50)
    print(f"  Epochs (max): {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Validation split: {validation_split*100:.0f}%")
    print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE} epochs")
    print("-"*50 + "\n")
    
    # Create callbacks
    callbacks = create_callbacks()
    
    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=verbose
    )
    
    # Get training statistics
    training_epochs = len(history.history['loss'])
    best_epoch = np.argmin(history.history['val_loss']) + 1
    best_val_loss = min(history.history['val_loss'])
    
    print("\n" + "-"*50)
    print("[Training] Training complete!")
    print(f"  Total epochs: {training_epochs}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print("-"*50 + "\n")
    
    # Final evaluation on test set
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"[Training] Test set performance:")
    print(f"  Test Loss (MSE): {test_loss:.4f}")
    print(f"  Test MAE: {test_mae:.4f} MPG")
    
    # Package results
    training_results = {
        'history': history.history,
        'epochs_trained': training_epochs,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'test_loss': test_loss,
        'test_mae': test_mae
    }
    
    return model, training_results


def get_training_summary(results: Dict) -> str:
    """
    Generate a human-readable training summary.
    
    Parameters
    ----------
    results : Dict
        Training results dictionary.
        
    Returns
    -------
    str
        Formatted training summary.
    """
    summary = f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    TRAINING SUMMARY                               ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  Epochs Trained:    {results['epochs_trained']:<8}                      ║
    ║  Best Epoch:        {results['best_epoch']:<8}                      ║
    ║  Best Val Loss:     {results['best_val_loss']:<8.4f}                      ║
    ║  Final Test MAE:    {results['test_mae']:<8.4f} MPG                 ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    return summary


if __name__ == "__main__":
    # Quick test when running module directly
    from data.loader import load_data
    from preprocessing.cleaner import clean_data
    from preprocessing.features import prepare_data_pipeline
    from models.neural_network import build_model
    
    raw_df = load_data()
    cleaned_df = clean_data(raw_df)
    data = prepare_data_pipeline(cleaned_df)
    
    # Build model
    input_dim = data['X_train'].shape[1]
    model = build_model(input_dim)
    
    # Train
    trained_model, results = train_neural_network(
        model,
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test'],
        verbose=1
    )
    
    print(get_training_summary(results))
