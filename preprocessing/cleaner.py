"""
Data Cleaning Module for Fuel Efficiency Prediction System.

This module handles all data cleaning operations including:
- Missing value imputation
- Data type conversions
- Categorical encoding
- Feature validation

CRITICAL INSIGHT ON PREPROCESSING:
---------------------------------
Poor preprocessing ruins regression performance MORE than model choice.

Why? Regression models (especially neural networks) are sensitive to:
1. Scale differences between features (weight ~3000 vs cylinders ~6)
2. Missing values propagating through calculations
3. Categorical variables treated as ordinal when they're nominal

A badly preprocessed dataset with XGBoost will perform WORSE than
a well-preprocessed dataset with linear regression. Data quality > Model complexity.
"""

import pandas as pd
import numpy as np
from utils.config import TARGET_COLUMN, FEATURE_COLUMNS


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    The Auto MPG dataset has missing values primarily in the 'horsepower'
    column (6 records out of 398). We have several options:
    
    1. Drop rows with missing values (loses 1.5% of data)
    2. Impute with mean/median (simple but ignores feature relationships)
    3. Impute with regression (uses other features to predict missing values)
    
    For this dataset, we use MEDIAN imputation because:
    - Horsepower has some outliers, so median is more robust than mean
    - The 6 missing records are too few to warrant complex imputation
    - Dropping would lose valuable training samples
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with potential missing values.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled.
    """
    df = df.copy()
    
    # Check for missing values before imputation
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print("[Cleaner] Missing values found:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"  - {col}: {count} missing ({count/len(df)*100:.1f}%)")
    
    # Impute horsepower with median
    if 'horsepower' in df.columns and df['horsepower'].isnull().any():
        median_hp = df['horsepower'].median()
        df['horsepower'] = df['horsepower'].fillna(median_hp)
        print(f"[Cleaner] Imputed horsepower with median value: {median_hp:.1f}")
    
    return df


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to appropriate data types.
    
    Proper data types are important because:
    - Numerical operations require numeric types
    - Categorical operations require category types
    - Memory usage can be reduced with appropriate types
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns to convert.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with converted types.
    """
    df = df.copy()
    
    # Ensure numerical columns are float
    numerical_cols = ['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Ensure integer columns are int
    integer_cols = ['cylinders', 'model_year', 'origin']
    for col in integer_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    return df


def encode_origin(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode the 'origin' categorical variable.
    
    Origin is a NOMINAL categorical variable (not ordinal):
    - 1 = USA
    - 2 = Europe  
    - 3 = Japan
    
    There's no inherent ordering between these values, so treating them
    as a single numeric feature would incorrectly imply that Japan (3) is
    "greater" than USA (1). One-hot encoding avoids this issue.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'origin' column.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with origin one-hot encoded.
    """
    df = df.copy()
    
    if 'origin' in df.columns:
        # Create one-hot encoded columns
        origin_dummies = pd.get_dummies(df['origin'], prefix='origin')
        
        # Rename for clarity
        origin_mapping = {
            'origin_1': 'origin_usa',
            'origin_2': 'origin_europe', 
            'origin_3': 'origin_japan'
        }
        origin_dummies = origin_dummies.rename(columns=origin_mapping)
        
        # Drop original column and concatenate dummies
        df = df.drop(columns=['origin'])
        df = pd.concat([df, origin_dummies], axis=1)
        
        print("[Cleaner] One-hot encoded 'origin' into: origin_usa, origin_europe, origin_japan")
    
    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that won't be used in modeling.
    
    The 'car_name' column contains the specific model name (e.g., 
    "chevrolet chevelle malibu"). While this could theoretically be
    used for lookup or brand-based features, it's:
    - High cardinality (305 unique values for 398 records)
    - Would cause severe overfitting if one-hot encoded
    - Better handled as brand extraction if needed later
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with potential unused columns.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with unused columns removed.
    """
    df = df.copy()
    
    columns_to_drop = ['car_name']
    existing_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    if existing_to_drop:
        df = df.drop(columns=existing_to_drop)
        print(f"[Cleaner] Dropped columns: {existing_to_drop}")
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full cleaning pipeline to raw dataset.
    
    This is the main entry point for data cleaning. It applies
    all cleaning steps in the correct order.
    
    Order matters:
    1. Drop unused columns first (reduces data to process)
    2. Handle missing values (before type conversion to avoid issues)
    3. Convert data types (ensures proper numeric/categorical handling)
    4. Encode categoricals (after type conversion for proper encoding)
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset from loader.
        
    Returns
    -------
    pd.DataFrame
        Cleaned dataset ready for feature engineering.
    """
    print("\n" + "="*50)
    print("Starting Data Cleaning Pipeline")
    print("="*50)
    
    df = drop_unused_columns(df)
    df = handle_missing_values(df)
    df = convert_data_types(df)
    df = encode_origin(df)
    
    print("\n" + "-"*50)
    print(f"[Cleaner] Cleaning complete: {len(df)} records, {len(df.columns)} features")
    print(f"[Cleaner] Remaining missing values: {df.isnull().sum().sum()}")
    print("-"*50 + "\n")
    
    return df


def validate_cleaned_data(df: pd.DataFrame) -> bool:
    """
    Validate that cleaned data meets quality requirements.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset.
        
    Returns
    -------
    bool
        True if all validations pass, raises exception otherwise.
    """
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        raise ValueError("Cleaned data still contains missing values!")
    
    # Check target variable exists and is numeric
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found!")
    
    if not pd.api.types.is_numeric_dtype(df[TARGET_COLUMN]):
        raise ValueError(f"Target column '{TARGET_COLUMN}' is not numeric!")
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if np.isinf(df[numeric_cols].values).any():
        raise ValueError("Cleaned data contains infinite values!")
    
    print("[Cleaner] Data validation passed ✓")
    return True


if __name__ == "__main__":
    # Quick test when running module directly
    from data.loader import load_data
    
    raw_df = load_data()
    cleaned_df = clean_data(raw_df)
    validate_cleaned_data(cleaned_df)
    
    print("\n--- Cleaned Data Preview ---")
    print(cleaned_df.head())
    print("\n--- Cleaned Data Info ---")
    print(cleaned_df.info())
