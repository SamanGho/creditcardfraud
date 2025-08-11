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

## 🎉 Success!
You've successfully run a complete fraud detection system with:
- Custom Decision Tree implementation
- No sklearn or ML libraries used
- Professional visualizations
- ~80% F1-Score on real fraud detection

---
For detailed documentation, see [README.md](README.md)
