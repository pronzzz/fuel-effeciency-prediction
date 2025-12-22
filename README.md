# 🚗 Fuel Efficiency Prediction System ⛽

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-green.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)](https://github.com/pronzzz/fuel-effeciency-prediction)

An **interpretable, robust regression system** for predicting vehicle fuel efficiency (MPG) using the Auto MPG dataset. This project goes beyond typical ML demos by emphasizing **understanding over accuracy** and treating regression as a **decision-support system**.

---

## 📋 Table of Contents

- [What Makes This Different](#-what-makes-this-different)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Key Insights](#-key-insights)
- [Generated Outputs](#-generated-outputs)
- [Detailed Guide](#-detailed-guide)
- [Future Extensions](#-future-extensions)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 What Makes This Different

| Typical Auto MPG Projects | **This Project** |
|---------------------------|------------------|
| Train one neural network | Compare **multiple models** for honest evaluation |
| Print a loss value | Explain *why* predictions behave as they do |
| Declare success | Demonstrate when NNs help—and when they **don't** |
| Black-box predictions | Prioritize **interpretability** over marginal gains |

> 💡 **Philosophy**: Numbers are only impressive when they explain something real.

---

## 📁 Project Structure

```
Fuel-efficiency-prediction/
├── 📂 data/
│   └── loader.py              # Data ingestion from UCI repository
├── 📂 preprocessing/
│   ├── cleaner.py             # Missing value handling, encoding
│   └── features.py            # Normalization, train-test split
├── 📂 eda/
│   └── visualizations.py      # Pairplots, correlation analysis
├── 📂 models/
│   ├── neural_network.py      # Keras Sequential model (64→32→1)
│   └── baselines.py           # Linear Regression, Ridge, Random Forest
├── 📂 training/
│   └── trainer.py             # Training loop with early stopping
├── 📂 evaluation/
│   ├── metrics.py             # MAE, RMSE, R² calculations
│   ├── plots.py               # Diagnostic visualizations
│   └── interpretability.py    # Feature importance analysis
├── 📂 utils/
│   └── config.py              # Hyperparameters and constants
├── 📂 outputs/                # Generated plots and artifacts
├── 🐍 main.py                 # End-to-end pipeline
├── 📄 requirements.txt
├── 📄 GUIDE.md                # Detailed usage guide
├── 📄 LICENSE
└── 📄 README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### 1. Clone the Repository

```bash
git clone https://github.com/pronzzz/fuel-effeciency-prediction.git
cd fuel-effeciency-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Full Pipeline

```bash
python main.py
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--eda-only` | Only run exploratory data analysis |
| `--skip-eda` | Skip EDA for faster execution |
| `--verbose` | Enable detailed output |

---

## 📊 Dataset

The **Auto MPG dataset** from the UCI Machine Learning Repository contains fuel efficiency data for automobiles from 1970-1982.

| Feature | Description | Type |
|---------|-------------|------|
| `mpg` | Miles per gallon | Target |
| `cylinders` | Number of cylinders | Discrete (3-8) |
| `displacement` | Engine displacement (cubic inches) | Continuous |
| `horsepower` | Engine horsepower | Continuous |
| `weight` | Vehicle weight (lbs) | Continuous |
| `acceleration` | 0-60 mph time (seconds) | Continuous |
| `model_year` | Year (70-82) | Discrete |
| `origin` | Country (USA/Europe/Japan) | Categorical |

**Dataset Statistics:**
- 📦 **398 samples**
- 🔢 **8 features** (7 predictors + 1 target)
- ❓ **6 missing values** (horsepower)

---

## 🧠 Model Architecture

### Neural Network

```
┌─────────────────────────────────────────────┐
│  INPUT (9 features after one-hot encoding) │
└─────────────────────┬───────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│  Dense(64, ReLU)                            │
│  Purpose: Learn feature combinations        │
└─────────────────────┬───────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│  Dense(32, ReLU)                            │
│  Purpose: Compress to efficiency-relevant   │
└─────────────────────┬───────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│  Dense(1, Linear)                           │
│  Purpose: Predict MPG                       │
└─────────────────────────────────────────────┘

Total Parameters: ~2,753
```

### Baselines for Comparison

| Model | Purpose |
|-------|---------|
| **Linear Regression** | Simple interpretable baseline |
| **Ridge Regression** | L2 regularization for multicollinearity |
| **Random Forest** | Nonlinear tree-based ensemble |

---

## 🔍 Key Insights

### 1️⃣ Weight Dominates MPG Prediction

**Vehicle weight** is the single strongest predictor of fuel efficiency:

- **Physics**: Heavier cars need more energy to accelerate (F = ma)
- **Correlation**: r ≈ -0.83 with MPG
- **Engineering insight**: 500 lb reduction → 3-5 MPG improvement

### 2️⃣ When Neural Networks Help (and When They Don't)

For tabular data with ~400 samples:

| Observation | Implication |
|-------------|-------------|
| NN provides modest improvement over linear regression | Complexity may not be worth it |
| Random Forest often matches NN performance | Simpler is often better |
| Baseline models are highly competitive | Start simple, add complexity only when needed |

### 3️⃣ Interpretability > Marginal Accuracy

In real-world engineering decisions:

> Knowing "reduce weight by 500 lbs → +3 MPG" is **more valuable** than knowing "predicted MPG is 32.7"

---

## 📈 Generated Outputs

After running the pipeline, check `outputs/` for:

| File | Description |
|------|-------------|
| `pairplot.png` | Feature relationships and correlations |
| `correlation_heatmap.png` | Multicollinearity detection |
| `target_distribution.png` | MPG distribution analysis |
| `features_by_origin.png` | Feature differences by car origin |
| `predicted_vs_actual_*.png` | Prediction accuracy per model |
| `residuals_*.png` | Error pattern analysis |
| `training_history.png` | Neural network convergence |
| `feature_importance_*.png` | Which features matter most |

---

## 📖 Detailed Guide

For a comprehensive walkthrough including:
- Step-by-step pipeline explanation
- Understanding each visualization
- Interpreting model results
- Customizing hyperparameters
- Extending the system

See **[GUIDE.md](GUIDE.md)**

---

## 🛠️ Future Extensions

- [ ] **SHAP values** for game-theoretic feature importance
- [ ] **Uncertainty estimation** for predictions
- [ ] **Scenario simulation** (e.g., "What if we reduce weight by 500 lbs?")
- [ ] **Gradient Boosting** (XGBoost, LightGBM) baselines
- [ ] **Cross-validation** for more robust evaluation
- [ ] **Web interface** for interactive predictions

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/auto+mpg) for the Auto MPG dataset
- Original dataset from StatLib library, Carnegie Mellon University

---

<p align="center">
  Made with ❤️ for interpretable machine learning
</p>
