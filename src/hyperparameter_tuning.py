"""
Hyperparameter tuning utilities for decision tree optimization.
"""

import itertools
import numpy as np
from typing import Dict, List, Tuple, Any
import time
from config import VERBOSITY_LEVEL
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("Warning: joblib not available, using sequential processing")


class HyperparameterTuner:
    """
    Grid search hyperparameter tuner for decision trees.
    """
    
    def __init__(self, decision_tree_class, verbosity=VERBOSITY_LEVEL):
        self.decision_tree_class = decision_tree_class
        self.verbosity = verbosity
        self.best_params = None
        self.best_score = -1
        self.tuning_results = []
    
    def grid_search(self, 
                   X_train: np.ndarray, 
                   y_train: np.ndarray,
                   X_val: np.ndarray, 
                   y_val: np.ndarray,
                   param_grid: Dict[str, List],
                   n_jobs: int = -1) -> Dict[str, Any]:
        """
        Perform parallelized grid search using validation set for hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training targets  
            X_val: Validation features
            y_val: Validation targets
            param_grid: Dictionary of parameter options to search
            n_jobs: Number of parallel jobs (-1 for all cores)
            
        Returns:
            Dictionary containing best parameters and results
        """
        if self.verbosity >= 1:
            print("\n" + "="*60)
            print("HYPERPARAMETER TUNING WITH VALIDATION SET (PARALLELIZED)")
            print("="*60)
            
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        total_combinations = len(param_combinations)
        if self.verbosity >= 1:
            print(f"Testing {total_combinations} parameter combinations...")
            if JOBLIB_AVAILABLE:
                import multiprocessing
                total_cores = multiprocessing.cpu_count()
                # Use 75% of cores for safety and system responsiveness
                n_cores = max(1, int(total_cores * 0.75))
                print(f"Detected {total_cores} cores, using {n_cores} cores for parallel processing")
            else:
                print("Using sequential processing (joblib not available)")
            
        def train_and_evaluate(param_combination):
            """Train and evaluate a single parameter combination."""
            params = dict(zip(param_names, param_combination))
            
            try:
                start_time = time.time()
                
                # Train model with these parameters
                model = self.decision_tree_class(**params)
                model.fit(X_train, y_train)
                
                # Evaluate on validation set
                val_predictions = model.predict(X_val)
                val_score = self._calculate_f1_score(y_val, val_predictions)
                
                training_time = time.time() - start_time
                
                return {
                    'params': params.copy(),
                    'val_f1_score': val_score,
                    'training_time': training_time,
                    'success': True
                }
                    
            except Exception as e:
                return {
                    'params': params.copy(),
                    'val_f1_score': 0.0,
                    'training_time': 0.0,
                    'error': str(e),
                    'success': False
                }
        
        # Check for minimum fraud cases in validation set
        fraud_count = np.sum(y_val == 1)
        if fraud_count < 5:
            print(f"Warning: Only {fraud_count} fraud cases in validation set. Results may be unstable.")
        
        # Execute parallel or sequential processing
        if JOBLIB_AVAILABLE and n_jobs != 1:
            # Parallel processing with computed cores
            import multiprocessing
            total_cores = multiprocessing.cpu_count()
            n_cores = max(1, int(total_cores * 0.75)) if n_jobs == -1 else min(n_jobs, total_cores)
            verbose_level = 10 if self.verbosity >= 2 else 0
            results = Parallel(n_jobs=n_cores, verbose=verbose_level)(
                delayed(train_and_evaluate)(param_combination) 
                for param_combination in param_combinations
            )
        else:
            # Sequential processing fallback
            results = []
            for i, param_combination in enumerate(param_combinations):
                result = train_and_evaluate(param_combination)
                results.append(result)
                
                if self.verbosity >= 2:
                    status = f"F1: {result['val_f1_score']:.4f}" if result['success'] else f"ERROR: {result.get('error', 'Unknown')}"
                    print(f"  {i+1:2d}/{total_combinations}: {result['params']} -> {status}")
        
        # Filter successful results and find best
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            raise RuntimeError("No parameter combinations succeeded during grid search")
        
        best_result = max(successful_results, key=lambda x: x['val_f1_score'])
        best_params = best_result['params']
        best_score = best_result['val_f1_score']
        
        # Store results
        self.best_params = best_params
        self.best_score = best_score
        self.tuning_results = successful_results
        
        if self.verbosity >= 1:
            print(f"\nHyperparameter tuning completed!")
            print(f"Successful parameter combinations: {len(successful_results)}/{total_combinations}")
            print(f"\nBest parameters found:")
            for param, value in best_params.items():
                print(f"  {param}: {value}")
            print(f"Best validation F1-score: {best_score:.4f}")
            
            # Show timing summary
            total_time = sum(r['training_time'] for r in successful_results)
            avg_time = total_time / len(successful_results) if successful_results else 0
            print(f"Total training time: {total_time:.2f}s, Average: {avg_time:.2f}s per combination")
            
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': successful_results
        }
    
    def _calculate_f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1-score for binary classification."""
        # Calculate precision and recall
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))
        
        if true_positives == 0:
            return 0.0
            
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        if precision + recall == 0:
            return 0.0
            
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def get_top_n_results(self, n: int = 5) -> List[Dict]:
        """Get top N parameter combinations by validation score."""
        if not self.tuning_results:
            return []
            
        sorted_results = sorted(self.tuning_results, 
                              key=lambda x: x['val_f1_score'], 
                              reverse=True)
        return sorted_results[:n]
    
    def print_tuning_summary(self):
        """Print a summary of hyperparameter tuning results."""
        if not self.tuning_results:
            print("No tuning results available.")
            return
            
        print("\n" + "="*50)
        print("HYPERPARAMETER TUNING SUMMARY")
        print("="*50)
        
        top_results = self.get_top_n_results(5)
        
        print("Top 5 parameter combinations:")
        for i, result in enumerate(top_results, 1):
            print(f"\n{i}. F1-Score: {result['val_f1_score']:.4f}")
            print(f"   Parameters: {result['params']}")
            print(f"   Training time: {result['training_time']:.3f}s")
            
        # Performance statistics
        scores = [r['val_f1_score'] for r in self.tuning_results]
        times = [r['training_time'] for r in self.tuning_results]
        
        print(f"\nPerformance Statistics:")
        print(f"  Best F1-Score: {max(scores):.4f}")
        print(f"  Average F1-Score: {np.mean(scores):.4f}")
        print(f"  Worst F1-Score: {min(scores):.4f}")
        print(f"  Average training time: {np.mean(times):.3f}s")


def create_param_grid(max_depths: List = None,
                     min_samples_splits: List = None, 
                     min_samples_leafs: List = None,
                     criteria: List = None) -> Dict[str, List]:
    """
    Create parameter grid for hyperparameter tuning.
    
    Args:
        max_depths: List of max_depth values to try
        min_samples_splits: List of min_samples_split values to try
        min_samples_leafs: List of min_samples_leaf values to try  
        criteria: List of criteria to try ('entropy', 'gini')
        
    Returns:
        Parameter grid dictionary
    """
    from config import (MAX_DEPTH_OPTIONS, MIN_SAMPLES_SPLIT_OPTIONS, 
                       MIN_SAMPLES_LEAF_OPTIONS)
    
    param_grid = {}
    
    if max_depths is not None:
        param_grid['max_depth'] = max_depths
    else:
        param_grid['max_depth'] = MAX_DEPTH_OPTIONS
        
    if min_samples_splits is not None:
        param_grid['min_samples_split'] = min_samples_splits  
    else:
        param_grid['min_samples_split'] = MIN_SAMPLES_SPLIT_OPTIONS
        
    if min_samples_leafs is not None:
        param_grid['min_samples_leaf'] = min_samples_leafs
    else:
        param_grid['min_samples_leaf'] = MIN_SAMPLES_LEAF_OPTIONS
        
    if criteria is not None:
        param_grid['criterion'] = criteria
    else:
        param_grid['criterion'] = ['entropy', 'gini']
        
    return param_grid
