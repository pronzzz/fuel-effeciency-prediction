"""
Exploratory Data Analysis (EDA) Visualizations for Fuel Efficiency Prediction.

This module provides visualization tools to understand the dataset before
modeling. Good EDA reveals:
- Feature correlations and multicollinearity
- Nonlinear relationships that might need special handling
- Outliers that could skew training
- The "story" behind the data

KEY INSIGHT: Understanding your data > fitting models blindly
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from utils.config import OUTPUT_DIR, FIGURE_DPI, FIGURE_SIZE, PAIRPLOT_VARS


def create_pairplot(df: pd.DataFrame, save: bool = True) -> None:
    """
    Create a pairplot to visualize feature relationships.
    
    Pairplots are the first tool for understanding feature interactions:
    - Diagonal shows distributions (look for skewness)
    - Off-diagonal shows scatter plots (look for correlations)
    
    Key patterns to look for in Auto MPG:
    ---------------------------------
    1. **Weight vs MPG**: Strong negative correlation (~-0.83)
       - Heavier cars = lower fuel efficiency
       - This is THE dominant relationship in the dataset
       - Physics: F=ma, heavier cars need more force to accelerate
    
    2. **Displacement vs MPG**: Negative correlation (~-0.80)
       - Larger engines = lower efficiency
       - Highly correlated with weight (large cars have large engines)
       
    3. **Horsepower vs MPG**: Negative correlation (~-0.78)
       - More powerful = less efficient
       - But notice: relationship is slightly nonlinear
       
    4. **Cylinders**: Discrete values (4, 5, 6, 8)
       - 4-cylinder cars cluster at high MPG
       - 8-cylinder cars cluster at low MPG
       
    5. **Acceleration**: Weak positive correlation
       - Slower acceleration (higher values) = slightly better MPG
       - But this is confounded by weight
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset with features and target.
    save : bool, optional
        Whether to save the plot to disk. Default True.
    """
    print("[EDA] Creating pairplot (this may take a moment)...")
    
    # Select columns for pairplot
    plot_cols = [col for col in PAIRPLOT_VARS if col in df.columns]
    
    # Set the style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Create pairplot with regression lines for off-diagonal
    g = sns.pairplot(
        df[plot_cols],
        diag_kind='kde',         # Kernel density estimate for diagonal
        corner=False,             # Show full matrix for complete view
        plot_kws={
            'alpha': 0.6,
            's': 30,
            'edgecolor': 'white',
            'linewidth': 0.5
        },
        diag_kws={
            'shade': True,
            'linewidth': 2
        }
    )
    
    g.fig.suptitle('Auto MPG Feature Relationships', y=1.02, fontsize=14)
    
    if save:
        filepath = os.path.join(OUTPUT_DIR, 'pairplot.png')
        g.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"[EDA] Pairplot saved to: {filepath}")
    
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Create a correlation heatmap to identify multicollinearity.
    
    Multicollinearity matters because:
    - Highly correlated features provide redundant information
    - Can cause unstable coefficients in linear models
    - Neural networks can handle it, but it's still useful knowledge
    
    In Auto MPG, we expect to see:
    - Weight, displacement, horsepower, cylinders all highly correlated
    - These all represent "car size" in different ways
    - Model year has moderate negative correlation with size features
      (cars got smaller/more efficient over time)
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with numerical features.
    save : bool, optional
        Whether to save the plot. Default True.
        
    Returns
    -------
    pd.DataFrame
        Correlation matrix for further analysis.
    """
    print("[EDA] Creating correlation heatmap...")
    
    # Calculate correlation matrix for numerical columns only
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Upper triangle mask
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={'shrink': 0.8}
    )
    
    ax.set_title('Feature Correlation Matrix\n(Lower Triangle)', fontsize=14, pad=20)
    
    if save:
        filepath = os.path.join(OUTPUT_DIR, 'correlation_heatmap.png')
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"[EDA] Correlation heatmap saved to: {filepath}")
    
    plt.close()
    
    return corr_matrix


def analyze_target_distribution(df: pd.DataFrame, save: bool = True) -> dict:
    """
    Analyze the distribution of the target variable (MPG).
    
    Understanding the target distribution helps:
    - Identify if transformation is needed (log, sqrt)
    - Spot data quality issues (impossible values)
    - Set realistic expectations for prediction errors
    
    Auto MPG target characteristics:
    - Range: ~10 to ~47 MPG
    - Mean: ~23 MPG
    - Slightly right-skewed (more low-MPG cars)
    - No extreme outliers that need removal
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'mpg' column.
    save : bool, optional
        Whether to save the plot. Default True.
        
    Returns
    -------
    dict
        Target statistics.
    """
    print("[EDA] Analyzing target distribution...")
    
    mpg = df['mpg']
    
    # Calculate statistics
    stats = {
        'mean': mpg.mean(),
        'median': mpg.median(),
        'std': mpg.std(),
        'min': mpg.min(),
        'max': mpg.max(),
        'skewness': mpg.skew(),
        'kurtosis': mpg.kurtosis()
    }
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram with KDE
    ax1 = axes[0]
    sns.histplot(mpg, kde=True, ax=ax1, color='steelblue', edgecolor='white')
    ax1.axvline(stats['mean'], color='red', linestyle='--', label=f"Mean: {stats['mean']:.1f}")
    ax1.axvline(stats['median'], color='green', linestyle='--', label=f"Median: {stats['median']:.1f}")
    ax1.set_xlabel('Miles Per Gallon (MPG)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distribution of Fuel Efficiency', fontsize=14)
    ax1.legend()
    
    # Box plot
    ax2 = axes[1]
    sns.boxplot(x=mpg, ax=ax2, color='steelblue')
    ax2.set_xlabel('Miles Per Gallon (MPG)', fontsize=12)
    ax2.set_title('MPG Box Plot', fontsize=14)
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(OUTPUT_DIR, 'target_distribution.png')
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"[EDA] Target distribution saved to: {filepath}")
    
    plt.close()
    
    # Print summary
    print("\n--- Target Variable (MPG) Statistics ---")
    for key, value in stats.items():
        print(f"  {key.capitalize()}: {value:.2f}")
    
    return stats


def analyze_features_by_origin(df: pd.DataFrame, save: bool = True) -> None:
    """
    Analyze how features differ by car origin.
    
    This reveals interesting patterns:
    - Japanese cars tend to be smaller and more efficient
    - American cars tend to be larger with more cylinders
    - European cars are in between
    
    Understanding these patterns helps interpret model predictions
    and explains why 'origin' is a useful feature.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with origin columns.
    save : bool, optional
        Whether to save the plot. Default True.
    """
    print("[EDA] Analyzing features by origin...")
    
    # Reconstruct origin column for visualization
    if 'origin_usa' in df.columns:
        df = df.copy()
        df['origin'] = 'Unknown'
        df.loc[df['origin_usa'] == 1, 'origin'] = 'USA'
        df.loc[df['origin_europe'] == 1, 'origin'] = 'Europe'
        df.loc[df['origin_japan'] == 1, 'origin'] = 'Japan'
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # MPG by origin
    sns.boxplot(data=df, x='origin', y='mpg', ax=axes[0, 0], palette='Set2')
    axes[0, 0].set_title('Fuel Efficiency by Origin')
    axes[0, 0].set_ylabel('MPG')
    
    # Weight by origin
    sns.boxplot(data=df, x='origin', y='weight', ax=axes[0, 1], palette='Set2')
    axes[0, 1].set_title('Vehicle Weight by Origin')
    axes[0, 1].set_ylabel('Weight (lbs)')
    
    # Horsepower by origin
    sns.boxplot(data=df, x='origin', y='horsepower', ax=axes[1, 0], palette='Set2')
    axes[1, 0].set_title('Horsepower by Origin')
    axes[1, 0].set_ylabel('Horsepower')
    
    # Cylinders by origin
    sns.countplot(data=df, x='cylinders', hue='origin', ax=axes[1, 1], palette='Set2')
    axes[1, 1].set_title('Cylinder Count by Origin')
    axes[1, 1].legend(title='Origin')
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(OUTPUT_DIR, 'features_by_origin.png')
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"[EDA] Features by origin saved to: {filepath}")
    
    plt.close()


def run_full_eda(df: pd.DataFrame) -> dict:
    """
    Run complete EDA pipeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset.
        
    Returns
    -------
    dict
        Dictionary containing EDA results and statistics.
    """
    print("\n" + "="*50)
    print("Starting Exploratory Data Analysis")
    print("="*50 + "\n")
    
    results = {}
    
    # Run all EDA functions
    create_pairplot(df)
    results['correlation_matrix'] = plot_correlation_heatmap(df)
    results['target_stats'] = analyze_target_distribution(df)
    analyze_features_by_origin(df)
    
    print("\n" + "-"*50)
    print("[EDA] Analysis complete. Check 'outputs/' for plots.")
    print("-"*50 + "\n")
    
    return results


if __name__ == "__main__":
    # Quick test when running module directly
    from data.loader import load_data
    from preprocessing.cleaner import clean_data
    
    raw_df = load_data()
    cleaned_df = clean_data(raw_df)
    results = run_full_eda(cleaned_df)
