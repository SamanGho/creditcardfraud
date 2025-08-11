"""
Complete PNG Visualization Generator
===================================

Generates all requested PNG visualizations for the fraud detection system:
1. tree_structure.png - Graphviz tree structure visualization
2. 1_dataset_distribution.png - Dataset class distribution
3. 2_model_performance_comparison.png - Model comparison chart
4. 3_confusion_matrix.png - Confusion matrix heatmap
5. 4_performance_metrics.png - Performance metrics bar chart
6. 5_technical_summary.png - Technical summary dashboard
7. 6_decision_tree_structure.png - Tree structure with Info Gain/Gini
8. complete_working_dashboard.png - Complete dashboard overview

Author: SAMANGHO
Date: 2025-08-10
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_tree_visualization import generate_tree_graphviz

def load_latest_model():
    """Load the latest saved model"""
    output_dir = Path("../outputs")
    if not output_dir.exists():
        print("⚠️  No outputs directory found. Please run the main system first.")
        return None, None, None
        
    # Look for models saved by the updated fraud_detection_main.py
    model_files = list(output_dir.glob("*_model.pkl"))
    if not model_files:
        # Fallback: check for old naming pattern
        model_files = list(output_dir.glob("best_model_*.pkl"))
        if not model_files:
            print("⚠️  No saved models found. Please run the main system first.")
            return None, None, None
    
    # Prefer quartile method if available, otherwise take the latest
    quartile_model = None
    for model_file in model_files:
        if 'quartile' in model_file.name.lower():
            quartile_model = model_file
            break
    
    latest_model_file = quartile_model or max(model_files, key=os.path.getctime)
    print(f"📁 Loading model: {latest_model_file.name}")
    
    try:
        with open(latest_model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        # Handle both old and new model save formats
        if isinstance(model_data, dict) and 'model' in model_data:
            # Old format with metadata
            model = model_data.get('model')
            metadata = model_data.get('metadata', {})
            feature_names = metadata.get('feature_names', [f'Feature_{i}' for i in range(30)])
        else:
            # New format - model object directly
            model = model_data
            metadata = {}
            feature_names = [f'V{i}' if i <= 28 else (['Time', 'Amount'][i-29]) for i in range(1, 31)]
        
        return model, metadata, feature_names
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None

# Import the  results extractor
from results_extractor import load_real_results

def create_real_data():
    """Load from trained models and actual dataset """
    print("📊 Loading  model results ")
    
    performance_data, confusion_matrices, dataset_info = load_real_results()
    
    if performance_data is None or not performance_data.get('method'):
        raise ValueError(
            "❌ NO  MODEL RESULTS AVAILABLE! "
            "Please run fraud_detection_main.py first to train models and generate  results. "
            
        )
    
    print(f"✅ Using  results from {len(performance_data['method'])} trained models")
    for i, method in enumerate(performance_data['method']):
        print(f"  {method}: F1={performance_data['f1_score'][i]:.4f} ")
    
    return performance_data, confusion_matrices, dataset_info

def generate_1_dataset_distribution():
    """Generate dataset distribution visualization"""
    print("📊 Generating 1_dataset_distribution.png...")
    
    _, _, dataset_info = create_real_data()
    
    plt.figure(figsize=(10, 6))
    
    # Create pie chart for class distribution
    plt.subplot(1, 2, 1)
    labels = ['Legitimate Transactions', 'Fraudulent Transactions']
    sizes = [dataset_info['normal_samples'], dataset_info['fraud_samples']]
    colors = ['lightblue', 'lightcoral']
    explode = (0, 0.1)  # explode fraud slice
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
            explode=explode, shadow=True, startangle=90)
    plt.title('Credit Card Transaction\nClass Distribution', fontweight='bold', fontsize=14)
    
    # Create bar chart for actual numbers
    plt.subplot(1, 2, 2)
    bars = plt.bar(labels, sizes, color=colors)
    plt.title('Transaction Counts', fontweight='bold', fontsize=14)
    plt.ylabel('Number of Transactions')
    plt.yscale('log')  # Log scale due to large difference
    
    # Add value labels on bars
    for bar, size in zip(bars, sizes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
                f'{size:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_path = "../outputs/1_dataset_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")

def generate_2_model_performance_comparison():
    """Generate model performance comparison chart"""
    print("📊 Generating 2_model_performance_comparison.png...")
    
    performance_data, _, _ = create_real_data()
    
    plt.figure(figsize=(12, 8))
    
    methods = performance_data['method']
    x = np.arange(len(methods))
    width = 0.2
    
    # Create grouped bar chart
    acc = performance_data['accuracy']
    prec = performance_data['precision']
    rec = performance_data['recall']
    f1 = performance_data['f1_score']
    plt.bar(x - width*1.5, acc, width, label='Accuracy', color='gold')
    plt.bar(x - width*0.5, prec, width, label='Precision', color='lightgreen')
    plt.bar(x + width*0.5, rec, width, label='Recall', color='lightblue')
    plt.bar(x + width*1.5, f1, width, label='F1-Score', color='lightcoral')
    
    plt.xlabel('Discretization Methods')
    plt.ylabel('Performance Score')
    plt.title('Model Performance Comparison Across\nDiscretization Methods', fontweight='bold', fontsize=14)
    plt.xticks(x, methods)
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Dynamic y-limits based on data
    ymin = max(0.0, min(min(acc), min(prec), min(rec), min(f1)) - 0.05)
    ymax = min(1.0, max(max(acc), max(prec), max(rec), max(f1)) + 0.05)
    plt.ylim(ymin, ymax)
    
    # Add value labels on bars
    for i, method in enumerate(methods):
        plt.text(i - width*1.5, acc[i] + 0.005, f"{acc[i]:.3f}", ha='center', va='bottom', fontsize=9)
        plt.text(i - width*0.5, prec[i] + 0.005, f"{prec[i]:.3f}", ha='center', va='bottom', fontsize=9)
        plt.text(i + width*0.5, rec[i] + 0.005, f"{rec[i]:.3f}", ha='center', va='bottom', fontsize=9)
        plt.text(i + width*1.5, f1[i] + 0.005, f"{f1[i]:.3f}", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_path = "../outputs/2_model_performance_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")

def generate_3_confusion_matrix():
    """Generate confusion matrix visualization"""
    print("📊 Generating 3_confusion_matrix.png...")
    
    _, confusion_matrices, _ = create_real_data()
    
    plt.figure(figsize=(15, 5))
    
    for i, (method, cm) in enumerate(confusion_matrices.items()):
        plt.subplot(1, 3, i + 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['Legitimate', 'Fraud'], 
                   yticklabels=['Legitimate', 'Fraud'])
        plt.title(f'{method} Method\nConfusion Matrix', fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
    
    plt.suptitle('Confusion Matrices for All Discretization Methods', 
                 fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    
    output_path = "../outputs/3_confusion_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")

def generate_4_performance_metrics():
    """Generate detailed performance metrics visualization"""
    print("📊 Generating 4_performance_metrics.png...")
    
    performance_data, confusion_matrices, _ = create_real_data()
    
    plt.figure(figsize=(16, 10))
    
    # Calculate additional metrics
    methods = performance_data['method']
    
    # Subplot 1: F1-Score comparison
    plt.subplot(2, 3, 1)
    f1_vals = performance_data['f1_score']
    bars = plt.bar(methods, f1_vals, color=['gold', 'silver', '#CD7F32'][:len(methods)])
    plt.title('F1-Score Comparison', fontweight='bold')
    plt.ylabel('F1-Score')
    ymin = max(0.0, min(f1_vals) - 0.05)
    ymax = min(1.0, max(f1_vals) + 0.05)
    plt.ylim(ymin, ymax)
    for bar, score in zip(bars, f1_vals):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Subplot 2: Precision vs Recall
    plt.subplot(2, 3, 2)
    plt.scatter(performance_data['recall'], performance_data['precision'], 
               s=200, c=['gold', 'silver', '#CD7F32'], alpha=0.7)
    for i, method in enumerate(methods):
        plt.annotate(method, (performance_data['recall'][i], performance_data['precision'][i]), 
                    xytext=(5, 5), textcoords='offset points', fontweight='bold')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: ROC points derived from confusion matrices ( single-threshold points)
    plt.subplot(2, 3, 3)
    fpr_points = []
    tpr_points = []
    labels = []
    for method in methods:
        cm = confusion_matrices[method]
        tn, fp, fn, tp = cm.ravel()
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr_points.append(fpr_val)
        tpr_points.append(tpr_val)
        labels.append(method)
    plt.scatter(fpr_points, tpr_points, s=120, c=['gold', 'silver', '#CD7F32'][:len(methods)], alpha=0.8)
    for i, label in enumerate(labels):
        plt.annotate(label, (fpr_points[i], tpr_points[i]), xytext=(5,5), textcoords='offset points')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Points (from Confusion Matrices)', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Detailed metrics table
    plt.subplot(2, 3, 4)
    plt.axis('off')
    table_data = []
    for i, method in enumerate(methods):
        cm = confusion_matrices[method]
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        table_data.append([
            method,
            f"{performance_data['accuracy'][i]:.3f}",
            f"{performance_data['precision'][i]:.3f}",
            f"{performance_data['recall'][i]:.3f}",
            f"{specificity:.3f}",
            f"{performance_data['f1_score'][i]:.3f}"
        ])
    
    table = plt.table(cellText=table_data,
                     colLabels=['Method', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    plt.title('Detailed Performance Metrics', fontweight='bold', y=0.9)
    
    # Subplot 5: Error analysis
    plt.subplot(2, 3, 5)
    false_positives = [cm[0, 1] for cm in confusion_matrices.values()]
    false_negatives = [cm[1, 0] for cm in confusion_matrices.values()]
    
    x = np.arange(len(methods))
    plt.bar(x - 0.2, false_positives, 0.4, label='False Positives', color='lightcoral')
    plt.bar(x + 0.2, false_negatives, 0.4, label='False Negatives', color='lightsalmon')
    plt.xlabel('Methods')
    plt.ylabel('Error Count')
    plt.title('Error Analysis', fontweight='bold')
    plt.xticks(x, methods, rotation=45)
    plt.legend()
    
    # Subplot 6: Performance summary
    plt.subplot(2, 3, 6)
    plt.axis('off')
    best_idx = int(np.argmax(performance_data['f1_score']))
    summary_text = f"""
🎯 PERFORMANCE SUMMARY 

✅ Best Overall Method: {methods[best_idx]}
   F1-Score: {performance_data['f1_score'][best_idx]:.3f}

📈 Highest Precision: {methods[int(np.argmax(performance_data['precision']))]}
   Precision: {max(performance_data['precision']):.3f}

🔍 Highest Recall: {methods[int(np.argmax(performance_data['recall']))]}
   Recall: {max(performance_data['recall']):.3f}

ℹ️ Metrics reflect the  trained models and dataset.
    """
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('Comprehensive Performance Metrics Analysis', 
                 fontweight='bold', fontsize=16, y=0.98)
    plt.tight_layout()
    
    output_path = "../outputs/4_performance_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")

def generate_5_technical_summary():
    """Generate technical summary dashboard a"""
    print("📊 Generating 5_technical_summary.png...")
    
    # Load  for accurate info
    performance_data, confusion_matrices, dataset_info = create_real_data()
    
    plt.figure(figsize=(16, 12))
    
    # Technical specifications (using dataset info)
    plt.subplot(3, 3, 1)
    plt.axis('off')
    tech_specs = f"""
🔧 TECHNICAL SPECIFICATIONS 

📊 Dataset: Credit Card Transactions
   Total Samples: {dataset_info['total_samples']:,}
   Fraud Samples: {dataset_info['fraud_samples']:,}
   Normal Samples: {dataset_info['normal_samples']:,}
   Classes: Binary (Fraud/Legitimate)
   Fraud Rate: {dataset_info['fraud_percentage']:.3f}%

🧠 Model: Decision Tree (From Scratch)
   Algorithm: Recursive Binary Splitting
   Criteria: Entropy & Gini Index
   Pruning: Post-pruning with F1 optimization
   Methods Trained: {len(performance_data['method'])}
    """
    plt.text(0.05, 0.95, tech_specs, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace')
    plt.title('System Specifications ', fontweight='bold', pad=20)
    
    # Real performance metrics
    plt.subplot(3, 3, 2)
    plt.axis('off')
    real_performance = f"""
📈 REAL MODEL PERFORMANCE

Best Method: {performance_data['method'][np.argmax(performance_data['f1_score'])]}
F1-Score: {max(performance_data['f1_score']):.4f}
Precision: {max(performance_data['precision']):.4f}
Recall: {max(performance_data['recall']):.4f}
Accuracy: {max(performance_data['accuracy']):.4f}

All Methods:
"""
    for i, method in enumerate(performance_data['method']):
        real_performance += f"{method}: {performance_data['f1_score'][i]:.4f}\n"
        
    plt.text(0.05, 0.95, real_performance, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace')
    plt.title('Real Performance Results', fontweight='bold', pad=20)
    
    # Implementation details (static but relevant)
    plt.subplot(3, 3, 3)
    plt.axis('off')
    implementation = """
⚙️ IMPLEMENTATION DETAILS

🏠 Architecture: Modular Design
   Core: decision_tree_model.py
   Tuning: hyperparameter_tuning.py
   Viz: complete_visualizations.py

🧪 Testing: Comprehensive Suite
   Unit Tests: Entropy, Gini, Info Gain
   Integration: Full system validation
   Performance: Cross-validation

🔄 Workflow: Complete Pipeline
   Data → Preprocess → Train → Prune → Evaluate
    """
    plt.text(0.05, 0.95, implementation, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace')
    plt.title('Implementation', fontweight='bold', pad=20)
    
    # Real F1-score comparison chart
    plt.subplot(3, 3, 4)
    methods = performance_data['method']
    f1_scores = performance_data['f1_score']
    
    bars = plt.bar(range(len(methods)), f1_scores, color=['gold', 'silver', '#CD7F32'][:len(methods)])
    plt.title('Real F1-Scores by Method', fontweight='bold')
    plt.ylabel('F1-Score')
    plt.xlabel('Method')
    plt.xticks(range(len(methods)), methods, rotation=45)
    
    # Add value labels
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{score:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.grid(True, alpha=0.3)
    
    # Real error analysis from confusion matrices
    plt.subplot(3, 3, 5)
    if confusion_matrices:
        error_data = []
        method_names = []
        for method, cm in confusion_matrices.items():
            tn, fp, fn, tp = cm.ravel()
            total_errors = fp + fn
            error_data.append(total_errors)
            method_names.append(method)
        
        bars = plt.bar(range(len(method_names)), error_data, color='lightcoral')
        plt.title('Real Total Errors by Method', fontweight='bold')
        plt.ylabel('Total Errors (FP + FN)')
        plt.xlabel('Method')
        plt.xticks(range(len(method_names)), method_names, rotation=45)
        
        for bar, errors in zip(bars, error_data):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{errors}', ha='center', va='bottom', fontsize=9)
    else:
        plt.text(0.5, 0.5, 'No confusion matrix data available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Error Analysis', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    
    # Real precision vs recall scatter
    plt.subplot(3, 3, 6)
    precision_vals = performance_data['precision']
    recall_vals = performance_data['recall']
    
    plt.scatter(recall_vals, precision_vals, s=100, c=['gold', 'silver', '#CD7F32'][:len(methods)], alpha=0.7)
    for i, method in enumerate(methods):
        plt.annotate(method, (recall_vals[i], precision_vals[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Real Precision vs Recall', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Dataset distribution pie chart
    plt.subplot(3, 3, 7)
    labels = ['Normal', 'Fraud']
    sizes = [dataset_info['normal_samples'], dataset_info['fraud_samples']]
    colors = ['lightblue', 'lightcoral']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.2f%%', startangle=90)
    plt.title(' Dataset Distribution', fontweight='bold')
    
    # Model comparison (accuracy)
    plt.subplot(3, 3, 8)
    accuracy_vals = performance_data['accuracy']
    
    bars = plt.bar(range(len(methods)), accuracy_vals, color='lightgreen')
    plt.title('Real Accuracy by Method', fontweight='bold')
    plt.ylabel('Accuracy')
    plt.xlabel('Method')
    plt.xticks(range(len(methods)), methods, rotation=45)
    
    for bar, acc in zip(bars, accuracy_vals):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{acc:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.grid(True, alpha=0.3)
    
    # Quality metrics (static but relevant)
    plt.subplot(3, 3, 9)
    plt.axis('off')
    quality = f"""
🎯 QUALITY METRICS

✅ Data Integrity:
   Real Models: {len(performance_data['method'])}

    """
    plt.text(0.05, 0.95, quality, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace')
    plt.title('Quality Assurance', fontweight='bold', pad=20)
    
    plt.suptitle('Technical Summary Dashboard ', 
                 fontweight='bold', fontsize=14, y=0.98)
    plt.tight_layout()
    
    output_path = "../outputs/5_technical_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path} ")

def generate_6_decision_tree_structure():
    """Generate decision tree structure visualization - ONLY from real trained models"""
    print("📊 Generating 6_decision_tree_structure.png...")
    
    model, metadata, feature_names = load_latest_model()
    
    if model is None:
        raise ValueError(
            "❌ NO TRAINED MODEL AVAILABLE FOR TREE STRUCTURE VISUALIZATION! "
            "Please run fraud_detection_main.py first to train models. "
            
        )
    
    # Create text-based tree structure from REAL model
    plt.figure(figsize=(16, 12))
    plt.axis('off')
    
    # Get tree structure as text from the REAL model
    if hasattr(model, 'visualize_tree'):
        # Redirect print output to capture tree structure
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            model.visualize_tree(max_depth=6)  # Show more depth for better detail
        tree_text = f.getvalue()
        
        if tree_text.strip():  # Only use if we got actual output
            plt.text(0.02, 0.98, tree_text, transform=plt.gca().transAxes, 
                    fontsize=9, verticalalignment='top', fontfamily='monospace')
        else:
            # Fallback: show model summary if visualize_tree doesn't work
            model_summary = f"""
🌳 REAL DECISION TREE MODEL SUMMARY 🌳
════════════════════════════════════════════════════════════════

⚠️  Tree visualization method not available for this model.
📊 However, this tree was trained from Dataset

🔢 Model Type: {type(model).__name__}
📁 Model Features: {len(feature_names)} features
🎯 Training Data: Real Credit Card Fraud Dataset


💡 To see detailed tree structure, ensure your model class
   implements the 'visualize_tree()' method.

📈 For more details, run the model training script to see
   the full tree output during training.
            """
            plt.text(0.02, 0.98, model_summary, transform=plt.gca().transAxes, 
                    fontsize=12, verticalalignment='top', fontfamily='monospace')
    else:
        # Model doesn't have visualize_tree method
        model_info = f"""
🌳  DECISION TREE MODEL LOADED 🌳
════════════════════════════════════════════════════════════════

✅ Successfully loaded trained model from dataset

🔢 Model Details:
   Type: {type(model).__name__}
   Features: {len(feature_names)} features
   Training:  Credit Card Fraud Dataset

⚠️  Tree structure visualization requires model.visualize_tree() method.

📊 This model contains  decision rules learned from training from data set i provided the link in ReadMe.md
💡 To see the tree structure:
   1. Ensure your DecisionTree class has visualize_tree() method
   2. Or check the console output when training
        """
        plt.text(0.02, 0.98, model_info, transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    plt.title(' Decision Tree Structure\n(From Trained Model - )', 
              fontweight='bold', fontsize=16, y=0.95, color='#2c3e50')
    
    output_path = "../outputs/6_decision_tree_structure.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path} (using  model structure)")

def generate_tree_structure_graphviz():
    """Generate tree structure using Graphviz if available"""
    print("🌳 Generating tree_structure.png with Graphviz...")
    
    model, metadata, feature_names = load_latest_model()
    
    if model is not None:
        # Try to generate Graphviz visualization
        try:
            output_path = generate_tree_graphviz(
                model, 
                feature_names, 
                output_dir="../outputs", 
                filename="tree_structure"
            )
            if output_path:
                print(f"✅ Graphviz visualization saved: {output_path}")
            else:
                raise RuntimeError("❌ Graphviz not available! Please install Graphviz to generate tree visualizations.")
        except Exception as e:
            raise RuntimeError(f"❌ Error generating tree visualization: {e}. Please ensure Graphviz is installed.")
    else:
        raise ValueError("❌ No trained model available for tree visualization!")


def generate_complete_working_dashboard():
    """Generate the complete working dashboard overview with corrected layout"""
    print("📊 Generating complete_working_dashboard.png...")
    
    performance_data, confusion_matrices, dataset_info = create_real_data()
    
    plt.figure(figsize=(24, 16))  # Wider figure for better spacing
    
    # Title
    plt.suptitle('🎯 COMPLETE FRAUD DETECTION SYSTEM DASHBOARD 🎯\n' +
                 'Credit Card Transaction Analysis with From-Scratch Decision Trees', 
                 fontweight='bold', fontsize=18, y=0.98)
    
    # Dataset Overview (Top Left)
    plt.subplot(4, 4, 1)
    labels = ['Legitimate', 'Fraud']
    sizes = [dataset_info['normal_samples'], dataset_info['fraud_samples']]
    colors = ['lightblue', 'lightcoral']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Dataset Distribution\n(284,807 transactions)', fontweight='bold')
    
    # Performance Comparison (Top Center)
    plt.subplot(4, 4, 2)
    methods = performance_data['method']
    plt.plot(methods, performance_data['f1_score'], 'o-', linewidth=3, markersize=8, color='gold')
    plt.ylabel('F1-Score')
    plt.title('Method Comparison\n(F1-Score)', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Confusion Matrix (Top Right)
    plt.subplot(4, 4, 3)
    best_idx = int(np.argmax(performance_data['f1_score']))
    best_method = methods[best_idx]
    best_cm = confusion_matrices[best_method]
    sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', cbar=False,
               xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
    plt.title(f'Best Model Confusion Matrix\n({best_method} Method)', fontweight='bold')
    
    # Key Metrics (Top Far Right)
    plt.subplot(4, 4, 4)
    plt.axis('off')
    best_method_idx = int(np.argmax(performance_data['f1_score']))
    best_method = performance_data['method'][best_method_idx]
    
    key_metrics = f"""
🏆 BEST MODEL RESULTS 

Method: {best_method}
F1-Score: {performance_data['f1_score'][best_method_idx]:.4f}
Precision: {performance_data['precision'][best_method_idx]:.4f}
Recall: {performance_data['recall'][best_method_idx]:.4f}
Accuracy: {performance_data['accuracy'][best_method_idx]:.4f}

🌳  Model Info:
Trained Methods: {len(performance_data['method'])}
    """
    plt.text(0.1, 0.9, key_metrics, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    plt.title(' Performance Metrics', fontweight='bold')
    
    # Detailed Performance Bars (Row 2, Span 2)
    plt.subplot(4, 4, (5, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    best_idx = int(np.argmax(performance_data['f1_score']))
    best_values = [
        performance_data['accuracy'][best_idx],
        performance_data['precision'][best_idx], 
        performance_data['recall'][best_idx],
        performance_data['f1_score'][best_idx]
    ]
    bars = plt.bar(metrics, best_values, color=['gold', 'lightgreen', 'lightblue', 'lightcoral'])
    plt.title(f'Detailed Performance Metrics - Best Model ({performance_data['method'][best_idx]})', fontweight='bold')
    plt.ylabel('Score')
    
    # Dynamic y-axis limits: start from 10% below the minimum value, but not below 0
    min_val = min(best_values)
    max_val = max(best_values)
    y_min = max(0, min_val - 0.1)  # Start from 10% below min, but not negative
    y_max = min(1.0, max_val + 0.05)  # Add 5% padding at top, but not above 1.0
    plt.ylim(y_min, y_max)
    
    # Add gridlines for better readability
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Position value labels above bars, adjusting for visibility
    for bar, value in zip(bars, best_values):
        # Calculate label position: above the bar with appropriate offset
        label_y = bar.get_height() + (y_max - y_min) * 0.02  # 2% of range above bar
        plt.text(bar.get_x() + bar.get_width()/2, label_y, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Tree Complexity (Row 2, Right) - Only if metadata available; otherwise skip
    plt.subplot(4, 4, 7)
    plt.axis('off')
    plt.title('Pruning Effectiveness (nodes)', fontweight='bold')
    try:
        model, _, feature_names = load_latest_model()
        if hasattr(model, 'get_tree_size') and hasattr(model, 'pruning_history'):
            before = getattr(model, 'pruning_history', {}).get('size_before', None)
            after = getattr(model, 'pruning_history', {}).get('size_after', None)
            if before is not None and after is not None:
                plt.axis('on')
                tree_data = ['Before', 'After']
                node_counts = [before, after]
                colors_tree = ['lightcoral', 'lightgreen']
                bars = plt.bar(tree_data, node_counts, color=colors_tree)
                plt.ylabel('Number of Nodes')
                for bar, count in zip(bars, node_counts):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{count}', ha='center', va='bottom', fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'Tree size metadata unavailable', ha='center', va='center')
    except Exception:
        plt.text(0.5, 0.5, 'Tree size metadata unavailable', ha='center', va='center')
    
    # Algorithm Details (Row 2, Far Right)
    plt.subplot(4, 4, 8)
    plt.axis('off')
    algorithm_details = """
⚙️ ALGORITHM SPECS

🧠 Decision Tree:
  • From-scratch implementation
  • Entropy & Gini criteria
  • Post-pruning with F1 optimization
  • Multi-core hyperparameter tuning

📊 Discretization:
  • Quartile binning (primary)
  • Equal width binning
  • Equal frequency binning

🔄 Pipeline:
  Data → Preprocess → Train → Prune → Evaluate
    """
    plt.text(0.05, 0.95, algorithm_details, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace')
    plt.title('Algorithm Details', fontweight='bold')
    
    # Method Comparison Chart (Row 3, Span 2)
    plt.subplot(4, 4, (9, 10))
    x = np.arange(len(methods))
    width = 0.2
    
    plt.bar(x - width, performance_data['accuracy'], width, label='Accuracy', color='gold', alpha=0.8)
    plt.bar(x, performance_data['precision'], width, label='Precision', color='lightgreen', alpha=0.8)
    plt.bar(x + width, performance_data['recall'], width, label='Recall', color='lightblue', alpha=0.8)
    
    plt.xlabel('Discretization Methods')
    plt.ylabel('Performance Score')
    plt.title('Comprehensive Method Comparison', fontweight='bold')
    plt.xticks(x, methods)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Dynamic y-axis to show all values properly
    all_values = list(performance_data['accuracy']) + list(performance_data['precision']) + list(performance_data['recall'])
    min_val = min(all_values)
    max_val = max(all_values)
    y_min = max(0, min_val - 0.1)  # Start from 10% below min, but not negative
    y_max = min(1.0, max_val + 0.05)  # Add 5% padding at top
    plt.ylim(y_min, y_max)
    
    # Feature Importance (Row 3, Right) - derive from model if available
    plt.subplot(4, 4, 11)
    plt.title('Feature Usage (Split Counts)')
    model, _, feature_names = load_latest_model()
    if model is not None and hasattr(model, 'get_feature_split_counts'):
        counts = model.get_feature_split_counts()
        if isinstance(counts, dict) and counts:
            # Map indices to names if needed
            labels = []
            values = []
            for k, v in counts.items():
                try:
                    idx = int(k)
                    labels.append(feature_names[idx] if idx < len(feature_names) else f'F{idx}')
                except:
                    labels.append(str(k))
                values.append(v)
            # Take top features
            order = np.argsort(values)[::-1][:6]
            labels = [labels[i] for i in order]
            values = [values[i] for i in order]
            plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        else:
            plt.axis('off')
            plt.text(0.5, 0.5, 'Feature usage unavailable', ha='center', va='center')
    else:
        plt.axis('off')
        plt.text(0.5, 0.5, 'Feature usage unavailable', ha='center', va='center')
    
    # System Performance (Row 3, Far Right) - remove hardcoded timings, show neutral info
    plt.subplot(4, 4, 12)
    plt.axis('off')
    system_perf = f"""
💻 SYSTEM INFO

Methods Trained: {len(performance_data['method'])}
Dataset Size: {dataset_info['total_samples']:,}
Fraud Rate: {dataset_info['fraud_percentage']:.3f}%

Note: Runtime and resource metrics are not displayed to avoid hardcoded values.
    """
    plt.text(0.05, 0.95, system_perf, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace')
    plt.title('System Information', fontweight='bold')
    
    # ROC points (Row 4, Left) derived from confusion matrices
    plt.subplot(4, 4, 13)
    fprs = []
    tprs = []
    for method in methods:
        cm = confusion_matrices[method]
        tn, fp, fn, tp = cm.ravel()
        fprs.append(fp / (fp + tn) if (fp + tn) else 0.0)
        tprs.append(tp / (tp + fn) if (tp + fn) else 0.0)
    plt.scatter(fprs, tprs, s=120)
    for i, method in enumerate(methods):
        plt.annotate(method, (fprs[i], tprs[i]), xytext=(5,5), textcoords='offset points')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Points (Confusion Matrix Derived)', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Error Analysis (Row 4, Center) - from best model confusion matrix
    plt.subplot(4, 4, 14)
    best_idx = int(np.argmax(performance_data['f1_score']))
    best_method = methods[best_idx]
    cm = confusion_matrices[best_method]
    tn, fp, fn, tp = cm.ravel()
    error_types = ['False\nPositives', 'False\nNegatives']
    errors = [int(fp), int(fn)]
    colors_error = ['lightcoral', 'lightsalmon']
    bars = plt.bar(error_types, errors, color=colors_error)
    plt.title(f'Error Analysis\n({best_method})', fontweight='bold')
    plt.ylabel('Count')
    
    for bar, error in zip(bars, errors):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{error}', ha='center', va='bottom', fontweight='bold')
    
    # Notes (Row 4, Right) - neutral, no hardcoded impact claims
    plt.subplot(4, 4, 15)
    plt.axis('off')
    notes = """
📝 NOTES

• Visuals reflect only  model outputs and dataset stats.
• No simulated or hardcoded performance numbers are shown.
• For deployment metrics (latency, resources), collect  runtime logs.
    """
    plt.text(0.05, 0.95, notes, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace')
    plt.title('Notes', fontweight='bold')
    
    # Project Summary (Row 4, Far Right) - data-driven summary
    plt.subplot(4, 4, 16)
    plt.axis('off')
    best_idx = int(np.argmax(performance_data['f1_score']))
    project_summary = f"""
📋 PROJECT SUMMARY 

• Methods evaluated: {', '.join(methods)}
• Best method by F1: {methods[best_idx]} ({performance_data['f1_score'][best_idx]:.3f})
• Accuracy range: {min(performance_data['accuracy']):.4f} - {max(performance_data['accuracy']):.4f}
• Precision range: {min(performance_data['precision']):.4f} - {max(performance_data['precision']):.4f}
• Recall range: {min(performance_data['recall']):.4f} - {max(performance_data['recall']):.4f}

Status: Visualizations reflect only  model outputs.
    """
    plt.text(0.05, 0.95, project_summary, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace')
    plt.title('Project Status', fontweight='bold')
    
    plt.tight_layout()
    
    # Add footer
# --- FIX: Move the footer text to the bottom-right to prevent overlap ---
    plt.figtext(0.99, 0.01,
            f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ha='right',
            fontsize=9,
            style='italic',
            color='#555555')
    
    output_path = "../outputs/complete_working_dashboard.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")

def generate_0_comprehensive_dashboard_fixed():
    """Generate the corrected comprehensive dashboard with proper spacing"""
    print("📊 Generating 0_comprehensive_dashboard.png (FIXED VERSION)...")
    
    performance_data, confusion_matrices, dataset_info = create_real_data()
    
    # Create large figure with better spacing
    fig = plt.figure(figsize=(24, 16))
    fig.suptitle('🌳 Complete Fraud Detection Analysis - Decision Tree Project', 
                 fontsize=24, fontweight='bold', y=0.98, color='#2c3e50')
    
    # Define beautiful color palette
    colors_main = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    # 1. Dataset Distribution (Top Left)
    ax1 = plt.subplot(3, 4, 1)
    legitimate_count = dataset_info['normal_samples']
    fraud_count = dataset_info['fraud_samples']
    labels = ['Legitimate', 'Fraud']
    sizes = [legitimate_count, fraud_count]
    colors_dist = ['#3498db', '#e74c3c']
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors_dist, 
                                       autopct='%1.2f%%', startangle=90,
                                       textprops={'fontsize': 12, 'fontweight': 'bold'},
                                       explode=(0.05, 0.1))  # Explode fraud slice
    
    ax1.set_title('Dataset Class Distribution\n(Credit Card Transactions)', 
                  fontsize=14, fontweight='bold', pad=20, color='#2c3e50')
    
    # Add beautiful text box with FIXED positioning
    textstr = f'Total: {dataset_info["total_samples"]:,} samples\nLegitimate: {legitimate_count:,}\nFraud: {fraud_count:,}\nImbalance: {fraud_count/legitimate_count*100:.3f}%'
    props = dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', alpha=0.9, edgecolor='#dee2e6')
    ax1.text(1.5, 0.5, textstr, transform=ax1.transAxes, fontsize=10,  # FIXED: Changed from 1.3 to 1.5
             verticalalignment='center', bbox=props, fontfamily='monospace')
    
    # 2. Model Performance Comparison (Top Center-Left)
    ax2 = plt.subplot(3, 4, 2)
    methods = performance_data['method']
    f1_scores = performance_data['f1_score']
    
    bars = ax2.bar(range(len(methods)), f1_scores, 
                   color=colors_main[:len(methods)], alpha=0.8, 
                   edgecolor='white', linewidth=2)
    
    ax2.set_title('Discretization Methods\nPerformance Comparison', 
                  fontsize=14, fontweight='bold', pad=20, color='#2c3e50')
    ax2.set_ylabel('F1-Score', fontweight='bold', fontsize=12, color='#2c3e50')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, fontsize=10, fontweight='bold', rotation=0)
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_facecolor('#fafafa')
    
    # Add value labels with better styling
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11, 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        # Add winner crown
        if score == max(f1_scores):
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.08,
                    '👑 Best', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold', color='#f39c12')
    
    # Add other panels with proper spacing
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.35, wspace=0.30)  # Increased wspace for better separation
    
    # Save with high quality
    output_path = "../outputs/0_comprehensive_dashboard.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ Saved CORRECTED dashboard: {output_path}")

def main():
    """Generate all requested PNG visualizations"""
    print("🎨 GENERATING ALL PNG VISUALIZATIONS")
    print("=" * 60)
    
    # Create outputs directory if it doesn't exist
    Path("../outputs").mkdir(exist_ok=True)
    
    # Set matplotlib style for consistent, professional visuals
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 10,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'figure.titleweight': 'bold'
    })
    
    try:
        # Generate all visualizations
        generate_0_comprehensive_dashboard_fixed() # 0_comprehensive_dashboard.png (CORRECTED VERSION)
        generate_tree_structure_graphviz()        # tree_structure.png (Enhanced Graphviz if available)
        generate_1_dataset_distribution()         # 1_dataset_distribution.png
        generate_2_model_performance_comparison()  # 2_model_performance_comparison.png
        generate_3_confusion_matrix()             # 3_confusion_matrix.png
        generate_4_performance_metrics()          # 4_performance_metrics.png
        generate_5_technical_summary()            # 5_technical_summary.png
        generate_6_decision_tree_structure()      # 6_decision_tree_structure.png
        generate_complete_working_dashboard()     # complete_working_dashboard.png
        
        print("\n✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print("📁 Check ../outputs/ directory for all PNG files")
        
        # List all generated files
        output_dir = Path("../outputs")
        png_files = list(output_dir.glob("*.png"))
        print(f"\n📊 Generated {len(png_files)} PNG visualizations:")
        for png_file in sorted(png_files):
            print(f"   📄 {png_file.name}")
            
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 PNG GENERATION COMPLETE! finally check out the images then go here https://youtu.be/DeumyOzKqgI ")
    else:
        print("\n⚠️  Some issues occurred during generation.")
