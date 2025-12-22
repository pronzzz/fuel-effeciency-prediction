"""
Neural Network Model for Fuel Efficiency Prediction.

This module defines the Keras Sequential model architecture for
predicting vehicle fuel efficiency (MPG).

ARCHITECTURE DESIGN RATIONALE:
-----------------------------
Why 64 -> 32 -> 1 hidden units?

1. **Dataset Size Consideration** (~400 samples)
   - Too few parameters = underfitting (can't capture patterns)
   - Too many parameters = overfitting (memorizes training data)
   - Rule of thumb: params << samples for good generalization
   
2. **Our architecture has ~3,000 parameters**
   - Input (9 features) -> Dense(64): 9*64 + 64 = 640 params
   - Dense(64) -> Dense(32): 64*32 + 32 = 2,080 params
   - Dense(32) -> Output(1): 32*1 + 1 = 33 params
   - Total: ~2,753 parameters
   - This is ~7 params per sample - reasonable for regression
   
3. **Gradual reduction (64 -> 32 -> 1)**
   - Allows hierarchical feature learning
   - First layer learns feature combinations
   - Second layer compresses to decision-relevant features
   - Output combines for final prediction

4. **Why ReLU activation?**
   - Non-linear: can model curved weight-MPG relationships
   - Computationally efficient (no exp() like sigmoid)
   - Reduces vanishing gradient problem
   - Standard choice for hidden layers

5. **Why Linear output?**
   - Regression target (MPG) is continuous
   - No bounds needed (though physically MPG > 0)
   - Sigmoid/tanh would squash output range
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils.config import (
    HIDDEN_LAYER_1_SIZE,
    HIDDEN_LAYER_2_SIZE,
    LEARNING_RATE
)


def build_model(input_dim: int) -> keras.Model:
    """
    Build a Sequential neural network for MPG regression.
    
    Architecture:
    - Input Layer: matches feature dimensions
    - Hidden Layer 1: 64 neurons, ReLU activation
    - Hidden Layer 2: 32 neurons, ReLU activation
    - Output Layer: 1 neuron, Linear activation (regression)
    
    Parameters
    ----------
    input_dim : int
        Number of input features.
        
    Returns
    -------
    keras.Model
        Compiled Keras model ready for training.
    """
    model = keras.Sequential([
        # Input layer specification
        layers.Input(shape=(input_dim,), name='input'),
        
        # First hidden layer
        # 64 neurons provide sufficient capacity for feature interaction learning
        layers.Dense(
            HIDDEN_LAYER_1_SIZE, 
            activation='relu',
            kernel_initializer='he_normal',  # Best for ReLU activation
            name='hidden_1'
        ),
        
        # Second hidden layer
        # Reduction to 32 neurons encourages learning compact representations
        layers.Dense(
            HIDDEN_LAYER_2_SIZE,
            activation='relu',
            kernel_initializer='he_normal',
            name='hidden_2'
        ),
        
        # Output layer
        # Single neuron with linear activation for regression
        # No activation = identity function = unbounded output
        layers.Dense(1, activation='linear', name='output')
    ])
    
    # Compile the model
    # - MSE loss: Standard for regression, penalizes large errors more
    # - Adam optimizer: Adaptive learning rate, works well out of the box
    # - MAE metric: More interpretable than MSE for monitoring
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',  # Mean Squared Error
        metrics=['mae']  # Mean Absolute Error for monitoring
    )
    
    return model


def get_model_summary(model: keras.Model) -> str:
    """
    Get a string summary of the model architecture.
    
    Parameters
    ----------
    model : keras.Model
        Compiled Keras model.
        
    Returns
    -------
    str
        Model summary as string.
    """
    # Capture summary
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    return '\n'.join(summary_lines)


def print_architecture_explanation() -> None:
    """
    Print explanation of the neural network architecture.
    
    This serves as documentation for users who want to understand
    the design decisions behind the model.
    """
    explanation = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║           NEURAL NETWORK ARCHITECTURE EXPLANATION                 ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║  INPUT (9 features)                                              ║
    ║      │                                                            ║
    ║      ▼                                                            ║
    ║  ┌─────────────────────────────────────────────────────────────┐ ║
    ║  │ HIDDEN LAYER 1: 64 neurons, ReLU                            │ ║
    ║  │ Purpose: Learn feature combinations (e.g., weight+power)    │ ║
    ║  └─────────────────────────────────────────────────────────────┘ ║
    ║      │                                                            ║
    ║      ▼                                                            ║
    ║  ┌─────────────────────────────────────────────────────────────┐ ║
    ║  │ HIDDEN LAYER 2: 32 neurons, ReLU                            │ ║
    ║  │ Purpose: Compress to efficiency-relevant representations    │ ║
    ║  └─────────────────────────────────────────────────────────────┘ ║
    ║      │                                                            ║
    ║      ▼                                                            ║
    ║  ┌─────────────────────────────────────────────────────────────┐ ║
    ║  │ OUTPUT: 1 neuron, Linear                                    │ ║
    ║  │ Purpose: Predict MPG (continuous value)                     │ ║
    ║  └─────────────────────────────────────────────────────────────┘ ║
    ║                                                                   ║
    ║  Total Parameters: ~2,753                                        ║
    ║  Parameters per sample: ~7 (good for generalization)            ║
    ║                                                                   ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(explanation)


if __name__ == "__main__":
    # Test model creation
    print("Testing neural network model creation...")
    
    # Build model for 9 features (after one-hot encoding origin)
    model = build_model(input_dim=9)
    
    print("\n--- Model Summary ---")
    model.summary()
    
    print_architecture_explanation()
