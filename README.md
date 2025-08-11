# 🚀 Quick Start Guide

Get the Credit Card Fraud Detection System running in 5 minutes!

## Prerequisites Check
```bash
python --version  # Should be 3.7 or higher
pip --version     # Should be installed
```

## Step 1: Setup (30 seconds)
```bash
# Clone and enter project
git clone <-https://github.com/SamanGho/creditcardfraud.git>
cd creditcardfraud

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Get Dataset (2 minutes)
1. Download from: https://www.kaggle.com/mlg-ulb/creditcardfraud
2. Place `creditcard.csv` in project root folder


## Step 3: Run Training (3 minutes)


```bash
cd src
python fraud_detection_main.py
```

Expected output:
```
🌳 COMPREHENSIVE FRAUD DETECTION SYSTEM 🌳
============================================================
Loading dataset...
Training models...
Best F1-Score: ~0.80
```

## Step 4: Generate Visualizations (30 seconds)
```bash
python generate_all_pngs.py
```

## 📁 Check Results
Look in the `outputs/` folder for:
- 9 PNG visualization files
- 3 trained model files (.pkl)

## 🎯 What You Just Did
✅ Trained 3 Decision Tree models from scratch  
✅ Compared 3 discretization methods  
✅ Generated comprehensive performance analysis  
✅ Created professional visualizations  

## 📊 Quick Performance Check
```python
# In Python interpreter
from results_extractor import load_real_results
performance, matrices, info = load_real_results()
print(f"Best F1-Score: {max(performance['f1_score']):.3f}")
```

## ⚡ Troubleshooting

### Dataset not found?
- Make sure `creditcard.csv` is in the project root (not in src/)

### ImportError?
```bash
pip install --upgrade -r requirements.txt
```

### Want better tree visualizations?
Install Graphviz: https://graphviz.org/download/


**Best Model**: Equal Frequency with 78.74% F1-Score and 96.15% Precision

## ✨ Key Features

### 1. Decision Tree Algorithm (`decision_tree_model.py`)
- **Custom Entropy**: `H(S) = -Σ(p_i * log2(p_i))`
- **Custom Gini Index**: `Gini(S) = 1 - Σ(p_i²)`
- **Information Gain**: `IG(S, A) = H(S) - Σ(|S_v|/|S| * H(S_v))`
- **Recursive Tree Building**: Binary splits with depth control
- **Post-Pruning**: F1-score optimization (269 → 37 nodes, 86% reduction)

### 2. Custom Metrics (`metrics.py`)
All implemented from scratch without sklearn:
```python
- calculate_accuracy()
- calculate_precision()
- calculate_recall()
- calculate_f1_score()
- calculate_confusion_matrix()
- calculate_roc_auc_score()  # Trapezoidal rule
- calculate_specificity()
- calculate_mcc()  # Matthews Correlation Coefficient
```

### 3. Data Preprocessing (`enhanced_data_preprocessor.py`)
- **Quartile Binning**: Primary method (required by specification)
- **Equal Width Binning**: Comparison method
- **Equal Frequency Binning**: Comparison method
- **Stratified Sampling**: Maintains fraud distribution
- **Information Value**: Feature importance calculation

### 4. Hyperparameter Tuning (`hyperparameter_tuning.py`)
- **Parallel Processing**: Uses 9 cores automatically
- **Grid Search**: 120 combinations for primary method
- **Parameters Tuned**:
  - max_depth: [5, 8, 10, 15, None]
  - min_samples_split: [2, 10, 50, 100]
  - min_samples_leaf: [1, 5, 10]
  - criterion: ['entropy', 'gini']

## ⚙️ Configuration

Edit `src/config.py` to customize:

```python
# Dataset Settings
USE_FULL_DATASET = True      # Use all 284,807 samples
SAMPLE_SIZE = 100000         # If not using full dataset

# Data Split (per requirements)
TRAIN_SIZE = 0.64   # 64% training (182,276 samples)
VAL_SIZE = 0.16     # 16% validation (45,569 samples)
TEST_SIZE = 0.20    # 20% testing (56,962 samples)

# Model Settings
PRIMARY_METHOD = 'quartile'
COMPARISON_METHODS = ['equal_width', 'equal_frequency']
HYPERPARAMETER_TUNING = True
VERBOSITY_LEVEL = 1  # 0=silent, 1=normal, 2=detailed
```

## 📝 Key Files

- **metrics.py**: All evaluation metrics from scratch
- **decision_tree_model.py**: Complete Decision Tree implementation
- **fraud_detection_main.py**: Main training pipeline
- **generate_all_pngs.py**: Visualization generator
- **example.py**: Interactive usage examples
- **config.py**: System configuration

## 🚫 What This Project Does NOT Use

- ❌ No sklearn
- ❌ No pre-built ML libraries
- ❌ No external metrics functions
- ❌ No tree building libraries
- ❌ No synthetic/fake data

Everything is implemented from scratch!

## 👥 Author

**SamanGho** - Complete implementation from scratch

## 📄 License

MIT License - See LICENSE file for details


## 🎉 Success!
You've successfully run a complete fraud detection system with:
- Custom Decision Tree implementation
- No sklearn or ML libraries used
- Professional visualizations
- ~80% F1-Score on real fraud detection

---
