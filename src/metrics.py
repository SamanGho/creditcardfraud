"""
Custom Evaluation Metrics (From Scratch)
========================================
Implements all evaluation metrics from scratch without using sklearn.
This ensures 100% compliance with the "from scratch" requirement.

Author: SamanGho
Date: 2025-08-11
"""

import numpy as np
from typing import Dict, Any, List, Tuple


def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> List[List[int]]:
    """
    Calculate confusion matrix from scratch for binary classification.
    
    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        
    Returns:
        2x2 confusion matrix as list of lists
        [[TN, FP],
         [FN, TP]]
    """
    # Ensure numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Calculate each component
    true_negatives = int(np.sum((y_true == 0) & (y_pred == 0)))
    false_positives = int(np.sum((y_true == 0) & (y_pred == 1)))
    false_negatives = int(np.sum((y_true == 1) & (y_pred == 0)))
    true_positives = int(np.sum((y_true == 1) & (y_pred == 1)))
    
    return [[true_negatives, false_positives], 
            [false_negatives, true_positives]]


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy from scratch.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy score (correct predictions / total predictions)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) == 0:
        return 0.0
    
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    
    return float(correct) / float(total)


def calculate_precision(y_true: np.ndarray, y_pred: np.ndarray, zero_division: float = 0.0) -> float:
    """
    Calculate precision from scratch.
    Precision = TP / (TP + FP)
    
    Args:
        y_true: True labels  
        y_pred: Predicted labels
        zero_division: Value to return when there are no positive predictions
        
    Returns:
        Precision score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    
    denominator = true_positives + false_positives
    
    if denominator == 0:
        return zero_division
    
    return float(true_positives) / float(denominator)


def calculate_recall(y_true: np.ndarray, y_pred: np.ndarray, zero_division: float = 0.0) -> float:
    """
    Calculate recall (sensitivity) from scratch.
    Recall = TP / (TP + FN)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        zero_division: Value to return when there are no actual positive cases
        
    Returns:
        Recall score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    
    denominator = true_positives + false_negatives
    
    if denominator == 0:
        return zero_division
    
    return float(true_positives) / float(denominator)


def calculate_f1_score(y_true: np.ndarray, y_pred: np.ndarray, zero_division: float = 0.0) -> float:
    """
    Calculate F1 score from scratch.
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        zero_division: Value to return when precision + recall = 0
        
    Returns:
        F1 score
    """
    precision = calculate_precision(y_true, y_pred, zero_division)
    recall = calculate_recall(y_true, y_pred, zero_division)
    
    denominator = precision + recall
    
    if denominator == 0:
        return zero_division
    
    return 2.0 * (precision * recall) / denominator


def calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray, zero_division: float = 0.0) -> float:
    """
    Calculate specificity from scratch.
    Specificity = TN / (TN + FP)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        zero_division: Value to return when there are no actual negative cases
        
    Returns:
        Specificity score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    true_negatives = np.sum((y_true == 0) & (y_pred == 0))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    
    denominator = true_negatives + false_positives
    
    if denominator == 0:
        return zero_division
    
    return float(true_negatives) / float(denominator)


def calculate_roc_auc_score(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Calculate ROC AUC score from scratch using the trapezoidal rule.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities or decision scores
        
    Returns:
        ROC AUC score
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    
    # Sort by scores in descending order
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    y_scores_sorted = y_scores[desc_score_indices]
    
    # Get unique scores and their indices
    distinct_value_indices = np.where(np.diff(y_scores_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    
    # Calculate TPR and FPR for each threshold
    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    
    # Total positives and negatives
    total_positives = np.sum(y_true == 1)
    total_negatives = np.sum(y_true == 0)
    
    if total_positives == 0 or total_negatives == 0:
        return 0.5  # Random classifier AUC
    
    # Calculate rates
    tpr = tps / total_positives
    fpr = fps / total_negatives
    
    # Add (0, 0) point for complete ROC curve
    tpr = np.r_[0, tpr]
    fpr = np.r_[0, fpr]
    
    # Calculate AUC using trapezoidal rule
    auc = np.trapz(tpr, fpr)
    
    return float(auc)


def calculate_comprehensive_metrics_from_scratch(y_true: np.ndarray, y_pred: np.ndarray, 
                                                 y_scores: np.ndarray = None) -> Dict[str, Any]:
    """
    Calculate all comprehensive metrics from scratch.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Optional predicted probabilities for ROC AUC
        
    Returns:
        Dictionary containing all metrics
    """
    # Calculate confusion matrix
    cm = calculate_confusion_matrix(y_true, y_pred)
    
    # Calculate all metrics
    metrics = {
        'confusion_matrix': cm,
        'accuracy': calculate_accuracy(y_true, y_pred),
        'precision': calculate_precision(y_true, y_pred, zero_division=0),
        'recall': calculate_recall(y_true, y_pred, zero_division=0),
        'f1_score': calculate_f1_score(y_true, y_pred, zero_division=0),
        'specificity': calculate_specificity(y_true, y_pred, zero_division=0)
    }
    
    # Add ROC AUC if scores are provided
    if y_scores is not None:
        metrics['roc_auc'] = calculate_roc_auc_score(y_true, y_scores)
    
    # Calculate additional useful metrics
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    
    # False Positive Rate (Fall-out)
    if (fp + tn) > 0:
        metrics['false_positive_rate'] = float(fp) / float(fp + tn)
    else:
        metrics['false_positive_rate'] = 0.0
    
    # False Negative Rate (Miss rate)
    if (fn + tp) > 0:
        metrics['false_negative_rate'] = float(fn) / float(fn + tp)
    else:
        metrics['false_negative_rate'] = 0.0
    
    # Matthews Correlation Coefficient
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator > 0:
        metrics['mcc'] = float(numerator) / float(denominator)
    else:
        metrics['mcc'] = 0.0
    
    # Balanced Accuracy
    metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2.0
    
    return metrics


# Alias functions to match sklearn naming conventions for easy replacement
accuracy_score = calculate_accuracy
precision_score = lambda y_true, y_pred, zero_division=0: calculate_precision(y_true, y_pred, zero_division)
recall_score = lambda y_true, y_pred, zero_division=0: calculate_recall(y_true, y_pred, zero_division)
f1_score = lambda y_true, y_pred, zero_division=0: calculate_f1_score(y_true, y_pred, zero_division)
confusion_matrix = lambda y_true, y_pred: np.array(calculate_confusion_matrix(y_true, y_pred))
roc_auc_score = calculate_roc_auc_score


def print_metrics_report(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray = None):
    """
    Print a comprehensive metrics report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Optional predicted probabilities
    """
    metrics = calculate_comprehensive_metrics_from_scratch(y_true, y_pred, y_scores)
    
    print("\n" + "="*60)
    print("COMPREHENSIVE METRICS REPORT (FROM SCRATCH)")
    print("="*60)
    
    print("\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"              Predicted")
    print(f"              Negative  Positive")
    print(f"Actual Negative  {cm[0][0]:6d}   {cm[0][1]:6d}")
    print(f"       Positive  {cm[1][0]:6d}   {cm[1][1]:6d}")
    
    print("\nPerformance Metrics:")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Precision:          {metrics['precision']:.4f}")
    print(f"  Recall (TPR):       {metrics['recall']:.4f}")
    print(f"  F1-Score:           {metrics['f1_score']:.4f}")
    print(f"  Specificity (TNR):  {metrics['specificity']:.4f}")
    
    if 'roc_auc' in metrics:
        print(f"  ROC AUC:            {metrics['roc_auc']:.4f}")
    
    print(f"\nAdditional Metrics:")
    print(f"  False Positive Rate: {metrics['false_positive_rate']:.4f}")
    print(f"  False Negative Rate: {metrics['false_negative_rate']:.4f}")
    print(f"  Balanced Accuracy:   {metrics['balanced_accuracy']:.4f}")
    print(f"  MCC:                 {metrics['mcc']:.4f}")
    print("="*60)


if __name__ == "__main__":
    # Test the metrics with sample data
    print("Testing custom metrics implementation...")
    
    # Sample data for testing
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.4, 0.2, 0.7, 0.8, 0.6])
    
    print_metrics_report(y_true, y_pred, y_scores)
    
    print("\n✅ All metrics calculated from scratch - no sklearn dependency :) finito !")
