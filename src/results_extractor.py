"""
Results Extractor for Real Model Performance Data
==================================================
Pulls real performance data from trained models
Author: SamanGho
Date: 2025-08-10
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from metrics import calculate_comprehensive_metrics_from_scratch
from typing import Dict, Any, Optional, Tuple


class RealResultsExtractor:
    """Extract real performance data from trained models  and dataset"""
    
    def __init__(self, outputs_dir: str = "../outputs"):
        self.outputs_dir = Path(outputs_dir)
        
    def load_real_dataset_info(self) -> Dict[str, Any]:
        """Load real dataset information from the  creditcard.csv file."""
        dataset_paths = [
            '../creditcard.csv',
            'creditcard.csv',
            '../data/creditcard.csv',
            'data/creditcard.csv'
        ]

        df = None
        for path in dataset_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                print(f"✅ Loaded real dataset from: {path}")
                break

        if df is None:
            raise FileNotFoundError("❌ Dataset not found! Please ensure creditcard.csv is available in the project dir if not get a kaggle and download it as i said.")

        fraud_samples = int(df['Class'].sum())
        total_samples = len(df)
        legitimate_samples = total_samples - fraud_samples

        return {
            'total_samples': total_samples,
            'fraud_samples': fraud_samples,
            'normal_samples': legitimate_samples,
            'fraud_percentage': (fraud_samples / total_samples) * 100
        }
    
    def extract_real_model_results(self) -> Tuple[Optional[Dict], Optional[Dict], Dict]:
        """
        Extract real performance results from saved trained models.
        
        Returns:
            Tuple of (performance_data, confusion_matrices, dataset_info)
        """
        print("📊 Extracting REAL model results from trained models...")
        
        # Load real dataset info
        dataset_info = self.load_real_dataset_info()
        
        # Find saved model files
        if not self.outputs_dir.exists():
            print("⚠️ No outputs directory found")
            return None, None, dataset_info
            
        model_files = list(self.outputs_dir.glob("*_model.pkl"))
        if not model_files:
            print("⚠️ No trained models found")
            return None, None, dataset_info
        
        print(f"✅ Found {len(model_files)} trained models")
        
        # Initialize result structures
        performance_data = {
            'method': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }
        confusion_matrices = {}
        
        # Map file names to display names
        method_mapping = {
            'quartile': 'Quartile',
            'equal_width': 'Equal Width',
            'equal_frequency': 'Equal Frequency'
        }
        
        # Load and evaluate each model
        for model_file in model_files:
            method_key = model_file.stem.replace('_model', '')
            method_name = method_mapping.get(method_key, method_key.title())
            
            try:
                print(f"📋 Processing model: {method_name}")
                
                # Load the model
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                
                # Evaluate using method-appropriate preprocessing to match training
                metrics, cm = self._evaluate_model_on_real_data(model, method_key)
                
                if metrics is not None:
                    performance_data['method'].append(method_name)
                    performance_data['accuracy'].append(metrics['accuracy'])
                    performance_data['precision'].append(metrics['precision'])
                    performance_data['recall'].append(metrics['recall'])
                    performance_data['f1_score'].append(metrics['f1_score'])
                    confusion_matrices[method_name] = cm
                    
                    print(f"✅ {method_name}: F1={metrics['f1_score']:.4f}, "
                          f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
                
            except Exception as e:
                print(f"❌ Error processing {method_name}: {e}")
                continue
        
        if not performance_data['method']:
            print("⚠️ No valid model results extracted")
            return None, None, dataset_info
        
        print(f"✅ Successfully extracted results for {len(performance_data['method'])} models")
        return performance_data, confusion_matrices, dataset_info
    
    def _evaluate_model_on_real_data(self, model, method_key: str) -> Tuple[Optional[Dict], Optional[np.ndarray]]:
        """
        Evaluate a trained model on real data to get legitimate performance metrics.
        method_key indicates which discretization to apply (to match training).
        """
        try:
            # Load the  dataset for evaluation
            dataset_paths = ['../creditcard.csv', 'creditcard.csv']
            df = None
            for path in dataset_paths:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    break
            
            if df is None:
                print("⚠️ Cannot evaluate - dataset not found is it there take a look miss spell?")
                return None, None
            
            # Use a test subset (last 20% of data) for evaluation
            test_size = int(0.2 * len(df))
            test_df = df.iloc[-test_size:].copy()
            
            # Prepare features and target
            X_test = test_df.drop('Class', axis=1).values
            y_test = test_df['Class'].values
            
            # Apply discretization to match training method
            X_test_processed = self._apply_method_preprocessing(X_test, method_key)
            
            # Make predictions
            y_pred = model.predict(X_test_processed)
            
            # Calculate metrics using the from scratch implementation
            metrics = calculate_comprehensive_metrics_from_scratch(y_test, y_pred)
            cm = np.array(metrics['confusion_matrix'])  # Convert back to numpy array if needed
            
            return metrics, cm
            
        except Exception as e:
            print(f"⚠️ Error in model evaluation: {e}")
            return None, None
    
    def _apply_method_preprocessing(self, X: np.ndarray, method_key: str) -> np.ndarray:
        """
        Apply discretization preprocessing that matches the training method key.
        Supported: quartile, equal_width, equal_frequency.
        """
        method_key = (method_key or '').lower()
        if 'quartile' in method_key:
            return self._discretize_quartile(X)
        if 'equal_width' in method_key:
            return self._discretize_equal_width(X)
        if 'equal_frequency' in method_key:
            return self._discretize_equal_frequency(X)
        # Fallback: quartile
        return self._discretize_quartile(X)
    
    def _discretize_quartile(self, X: np.ndarray) -> np.ndarray:
        X_discretized = np.zeros_like(X)
        for i in range(X.shape[1]):
            feature = X[:, i]
            q1, q2, q3 = np.percentile(feature, [25, 50, 75])
            X_discretized[:, i] = np.select([
                feature <= q1,
                (feature > q1) & (feature <= q2),
                (feature > q2) & (feature <= q3),
                feature > q3
            ], [0, 1, 2, 3])
        return X_discretized.astype(int)
    
    def _discretize_equal_width(self, X: np.ndarray, n_bins: int = 4) -> np.ndarray:
        X_discretized = np.zeros_like(X)
        for i in range(X.shape[1]):
            feature = X[:, i]
            min_val, max_val = np.min(feature), np.max(feature)
            if min_val == max_val:
                X_discretized[:, i] = 0
            else:
                width = (max_val - min_val) / n_bins
                bins = [min_val + width * j for j in range(n_bins + 1)]
                bins[-1] += 1e-9  # include max
                X_discretized[:, i] = np.digitize(feature, bins) - 1
        return np.clip(X_discretized, 0, n_bins - 1).astype(int)
    
    def _discretize_equal_frequency(self, X: np.ndarray, n_bins: int = 4) -> np.ndarray:
        X_discretized = np.zeros_like(X)
        for i in range(X.shape[1]):
            feature = X[:, i]
            percentiles = [100 * j / n_bins for j in range(1, n_bins)]
            thresholds = np.percentile(feature, percentiles)
            X_discretized[:, i] = np.zeros(len(feature))
            for threshold in thresholds:
                X_discretized[:, i] += (feature > threshold)
        return X_discretized.astype(int)


def load_real_results() -> Tuple[Optional[Dict], Optional[Dict], Dict]:
    """
    Main function to load real model results.
    
    Returns:
        Tuple of (performance_data, confusion_matrices, dataset_info)
    """
    extractor = RealResultsExtractor()
    return extractor.extract_real_model_results()


if __name__ == "__main__":
    # Test the extractor
    performance_data, confusion_matrices, dataset_info = load_real_results()
    
    print("\n🔍 EXTRACTION RESULTS:")
    print(f"Dataset Info: {dataset_info}")
    if performance_data:
        print(f"Performance Data: {len(performance_data['method'])} methods")
        for i, method in enumerate(performance_data['method']):
            print(f"  {method}: F1={performance_data['f1_score'][i]:.4f}")
    if confusion_matrices:
        print(f"Confusion Matrices: {len(confusion_matrices)} available")
