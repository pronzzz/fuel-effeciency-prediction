"""
Configuration module for Fuel Efficiency Prediction System.

This module centralizes all hyperparameters, constants, and configuration
settings to ensure reproducibility and easy experimentation.
"""

import os

# =============================================================================
# REPRODUCIBILITY SETTINGS
# =============================================================================
# Using a fixed random seed ensures that results are reproducible across runs.
# This is crucial for scientific validity and debugging.
RANDOM_SEED = 42

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
# Auto MPG dataset from UCI Machine Learning Repository
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"

# Column names for the Auto MPG dataset (not included in raw data)
COLUMN_NAMES = [
    'mpg',           # Target variable: Miles per gallon
    'cylinders',     # Number of cylinders (4, 6, 8)
    'displacement',  # Engine displacement in cubic inches
    'horsepower',    # Engine horsepower
    'weight',        # Vehicle weight in pounds
    'acceleration',  # Time to accelerate from 0 to 60 mph (seconds)
    'model_year',    # Model year (70-82, representing 1970-1982)
    'origin',        # Origin of car (1=USA, 2=Europe, 3=Japan)
    'car_name'       # Car name (will be dropped - not useful for prediction)
]

TARGET_COLUMN = 'mpg'
FEATURE_COLUMNS = ['cylinders', 'displacement', 'horsepower', 'weight', 
                   'acceleration', 'model_year', 'origin']

# =============================================================================
# DATA SPLITTING
# =============================================================================
# 80-20 split is a reasonable default for this dataset size (~400 samples)
# Larger test sets would reduce training data too much
TEST_SPLIT_RATIO = 0.2

# =============================================================================
# NEURAL NETWORK HYPERPARAMETERS
# =============================================================================
# Architecture: Input -> 64 -> 32 -> 1
# Rationale:
# - 64 neurons: Provides sufficient capacity for learning complex feature interactions
# - 32 neurons: Gradual reduction prevents bottleneck while encouraging compression
# - For ~400 samples, this is enough capacity without severe overfitting risk

HIDDEN_LAYER_1_SIZE = 64
HIDDEN_LAYER_2_SIZE = 32

# Training hyperparameters
LEARNING_RATE = 0.001      # Adam's default, works well for most problems
BATCH_SIZE = 32            # Small enough for smooth gradients, large enough for efficiency
EPOCHS = 500               # High limit - early stopping will handle actual termination

# Early stopping configuration
# Patience of 20 epochs allows model to escape local plateaus while preventing waste
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 0.0001  # Minimum improvement to count as progress

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================
FIGURE_DPI = 150
FIGURE_SIZE = (10, 8)
PAIRPLOT_VARS = ['mpg', 'cylinders', 'displacement', 'horsepower', 
                 'weight', 'acceleration']
