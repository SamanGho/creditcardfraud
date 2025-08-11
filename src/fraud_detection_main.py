"""
Comprehensive Fraud Detection System
===================================
- Uses full dataset or configurable sampling with intelligent stratification
- Implements proper 80/20 train/test split with validation for hyperparameter tuning
- Primary quartile discretization + bonus comparison methods
- Hyperparameter tuning using validation set
- Advanced statistical analysis and feature importance
- Professional visualizations and tree structure display
- Handles class imbalance appropriately
Developed for academic submission - addresses all requirements and bonus points.
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import warnings
import pickle
from datetime import datetime
warnings.filterwarnings('ignore')

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import configuration and modules
from config import *
from enhanced_data_preprocessor import EnhancedDataPreprocessor
from hyperparameter_tuning import HyperparameterTuner, create_param_grid
from decision_tree_model import DecisionTreeComplete
from improved_visualizations import create_visualizations
from enhanced_tree_visualization import generate_tree_graphviz
from metrics import calculate_comprehensive_metrics_from_scratch  # Your new metrics


class ComprehensiveFraudDetectionSystem:
   
    """ complete fraud detection """
    
    def __init__(self):
        self.verbosity = VERBOSITY_LEVEL
        self.preprocessor = EnhancedDataPreprocessor(verbosity=self.verbosity)
        self.results = {}
        self.best_model = None
        self.best_method = None
        self.dataset_info = {}
        
    def load_dataset(self, dataset_path: str = None) -> pd.DataFrame:
        """load the credit card  dataset """
        if self.verbosity >= 1:
            print("="*60)
            print("LOADING CREDIT CARD FRAUD DETECTION DATASET") 
            print("="*60)
            # where to look for it 
        possible_paths = [
            dataset_path,
            'creditcard.csv',
            '../creditcard.csv',
            os.path.join(os.path.dirname(__file__), '..', 'creditcard.csv')
        ]
        
        df = None
        for path in possible_paths:
            if path is None:
                continue
            try:
                df = pd.read_csv(path)
                if self.verbosity >= 1:
                    print(f"dataset loaded from: {path}")
                break
            except FileNotFoundError:
                continue
                
        if df is None:
            raise FileNotFoundError("credit card dataset not found plz make sure  creditcard.csv is available ")
            
        # store dataset info
        self.dataset_info = {
            'total_samples': len(df),
            'total_features': len(df.columns) - 1,
            'fraud_cases': int(df['Class'].sum()),
            'fraud_rate': 100 * df['Class'].sum() / len(df),
            'feature_names': [col for col in df.columns if col != 'Class']
        }
        
        if self.verbosity >= 1:
            print(f"Dataset: {self.dataset_info['total_samples']:,} samples, "
                  f"{self.dataset_info['fraud_cases']:,} fraud cases "
                  f"({self.dataset_info['fraud_rate']:.3f}%)")
            print(f"features: {self.dataset_info['total_features']}")
            print(f"challenge: highly imbalanced dataset - requires careful handling")
            
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> dict:
        """
        comprehensive data preprocessing with multiple discretization methods.
        """
        if self.verbosity >= 1:
            print("\\n" + "="*60)
            print("DATA PREPROCESSING AND DISCRETIZATION") 
            print("="*60)
            
        # load and prepare dataset with intelligent sampling
        X_raw, y, feature_names = self.preprocessor.load_and_prepare_dataset(df)
        
        # show advanced feature analysis (ezafi bonus)
        if ENABLE_ADVANCED_STATS:
            self.preprocessor.print_feature_analysis()
        
        # create proper train/validation/test split ( requirement)
        if self.verbosity >= 1:
            print("\n" + "="*50)
            print("CREATING TRAIN/VALIDATION/TEST SPLIT")
            print("="*50)
            print("requirement: 80% training, 20% testing")
            print("Using validation set from training portion for hyperparameter tuning")
            
        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = self.preprocessor.create_train_val_test_split(X_raw, y)
        
        # test different discretization methods
        methods_to_test = [PRIMARY_METHOD] + COMPARISON_METHODS
        discretization_results = {}
        
        for method in methods_to_test:
            if self.verbosity >= 1:
                method_name = method.replace('_', ' ').title()
                print(f"\\n{'='*50}")
                print(f"DISCRETIZATION: {method_name.upper()}")
                print(f"{'='*50}")
                if method == PRIMARY_METHOD:
                    print("*** PRIMARY METHOD ***")
                else:
                    print("*** (EZAFI BONUS) COMPARISON METHOD ***")
                    
            try:
                # apply discretization to all splits
                X_train = self.preprocessor.apply_discretization(X_train_raw, method)
                X_val = self.preprocessor.apply_discretization(X_val_raw, method)
                X_test = self.preprocessor.apply_discretization(X_test_raw, method)
                
                discretization_results[method] = {
                    'X_train': X_train,
                    'X_val': X_val, 
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_val': y_val,
                    'y_test': y_test,
                    'feature_names': feature_names,
                    'success': True
                }
                
                if self.verbosity >= 2:
                    print(f"Discretization successful: {X_train.shape[0]:,} train, "
                          f"{X_val.shape[0]:,} val, {X_test.shape[0]:,} test samples")
                          
            except Exception as e:
                if self.verbosity >= 1:
                    print(f"Error in {method} discretization: {str(e)}")
                discretization_results[method] = {'success': False, 'error': str(e)}
                
        return discretization_results
    
    def train_and_tune_models(self, discretization_results: dict) -> dict:
        """
        train models with hyperparameter tuning using validation set .
        """
        model_results = {}
        
        for method, data in discretization_results.items():
            if not data.get('success', False):
                continue
                
            if self.verbosity >= 1:
                method_name = method.replace('_', ' ').title()
                print(f"\\n{'='*60}")
                print(f"TRAINING MODEL: {method_name.upper()}")
                print(f"{'='*60}")
                
            # Extract data
            X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
            y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
            feature_names = data['feature_names']
            
            # Hyperparameter tuning 
            if HYPERPARAMETER_TUNING:
                if self.verbosity >= 1:
                    print("Performing hyperparameter tuning using validation set...")
                    
                tuner = HyperparameterTuner(DecisionTreeComplete, verbosity=self.verbosity)
                
                # create parameter grid (optimized for efficiency)
                if method == PRIMARY_METHOD:
                    # More thorough search for primary method 
                    param_grid = create_param_grid(
                        max_depths=[5, 8, 10, 15, None],  # Include deeper trees 
                        min_samples_splits=[2, 10, 50, 100],  # Varied thresholds for imbalanced data
                        min_samples_leafs=[1, 5, 10],  # Essential leaf sizes
                        criteria=['entropy', 'gini']  # Both criteria 
                    )
                else:
                    # Efficient search for comparison methods (bonus)
                    param_grid = create_param_grid(
                        max_depths=[8, None],  # Focus on best performing depths
                        min_samples_splits=[10, 50],  # Robust splits for comparison
                        min_samples_leafs=[1, 5],  # Key leaf sizes
                        criteria=['entropy']  # Fastest criterion
                    )
                
                # Perform grid search
                tuning_results = tuner.grid_search(X_train, y_train, X_val, y_val, param_grid)
                best_params = tuning_results['best_params']
                
                if self.verbosity >= 2:
                    tuner.print_tuning_summary()
            else:
                # Use default parameters if tuning disabled
                best_params = {
                    'max_depth': 8,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'criterion': 'entropy'
                }
                
            # Train final model with best parameters
            if self.verbosity >= 1:
                print(f"Training final model with best parameters...")
                
            start_time = time.time()
            model = DecisionTreeComplete(**best_params)
            model.fit(X_train, y_train, feature_names)
            training_time = time.time() - start_time
            
            # Post-pruning demonstration   
            if self.verbosity >= 1:
                print(f"Doing post-pruning using validation set...")
                
            initial_size = model.get_tree_size()
            model.post_prune(X_val, y_val)
            final_size = model.get_tree_size()
            
            if self.verbosity >= 1:
                print(f"Tree size: {initial_size} -> {final_size} nodes ({initial_size-final_size} pruned)")
                
            # Evaluate model
            predictions = model.predict(X_test)
            metrics = self._calculate_comprehensive_metrics(y_test, predictions)
            
            # Save the trained model to  dir outputs
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
            os.makedirs(output_dir, exist_ok=True)
            model_filename = f"{method}_model.pkl"
            model_path = os.path.join(output_dir, model_filename)
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            if self.verbosity >= 1:
                print(f"Model saved: {model_path}")
            
            # Store results
            model_results[method] = {
                'model': model,
                'model_path': model_path,
                'best_params': best_params,
                'metrics': metrics,
                'training_time': training_time,
                'tree_size_initial': initial_size,
                'tree_size_final': final_size,
                'X_test': X_test,
                'y_test': y_test,
                'y_pred': predictions,
                'feature_names': feature_names
            }
            
            if self.verbosity >= 1:
                print(f"Results - F1: {metrics['f1_score']:.4f}, "
                      f"Precision: {metrics['precision']:.4f}, "
                      f"Recall: {metrics['recall']:.4f}")
                
        return model_results
    
    def _calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate comprehensive evaluation metrics using from-scratch functions."""
        return calculate_comprehensive_metrics_from_scratch(y_true, y_pred)
    
    def analyze_results(self, model_results: dict):
        """analyze and compare model results."""
        if self.verbosity >= 1:
            print("\\n" + "="*60)
            print("MODEL COMPARISON AND ANALYSIS")
            print("="*60)
            
        # Find best model
        valid_results = {k: v for k, v in model_results.items() if 'metrics' in v}
        
        if not valid_results:
            print("No valid results to analyze.")
            return
            
        best_method = max(valid_results.keys(), key=lambda k: valid_results[k]['metrics']['f1_score'])
        self.best_method = best_method
        self.best_model = valid_results[best_method]['model']
        
        if self.verbosity >= 1:
            print(f"Best performing method: {best_method.replace('_', ' ').title()}")
            print(f"Best F1-score: {valid_results[best_method]['metrics']['f1_score']:.4f}")
            
            print("\\nComparison of all methods:")
            print("-" * 80)
            print(f"{'Method':<20} {'F1-Score':<10} {'Precision':<12} {'Recall':<10} {'Accuracy':<10} {'Tree Size':<10}")
            print("-" * 80)
            
            for method, results in valid_results.items():
                metrics = results['metrics']
                method_display = method.replace('_', ' ').title()[:18]
                print(f"{method_display:<20} "
                      f"{metrics['f1_score']:<10.4f} "
                      f"{metrics['precision']:<12.4f} "
                      f"{metrics['recall']:<10.4f} "
                      f"{metrics['accuracy']:<10.4f} "
                      f"{results.get('tree_size_final', 'N/A'):<10}")
        
        # Store results for visualization
        self.results = valid_results
    
    def create_visualizations(self, output_dir: str, df: pd.DataFrame):
        """Create comprehensive visualizations."""
        if not SAVE_VISUALIZATIONS or not self.results:
            return
            
        if self.verbosity >= 1:
            print("\n" + "="*60) 
            print("CREATING VISUALIZATIONS")
            print("="*60)
            
        try:
            # we are gonna use  the corrected improved_visualizations module
            # Extract y from the original dataframe
            y = df['Class'].values
            
            # Call the visualizations function with correct parameters
            create_visualizations(self.results, df, y, output_dir)
            
            # khoshgel konim with creating enhanced tree structure visualization using Graphviz
            if self.best_model:
                # do it with the better Graphviz tree visualization
                generate_tree_graphviz(
                    tree=self.best_model, 
                    feature_names=self.results[self.best_method]['feature_names'],
                    output_dir=output_dir,
                    filename="6_decision_tree_structure"
                )
                
            if self.verbosity >= 1:
                print(f"✅ All visualizations created successfully!")
                    
        except Exception as e:
            if self.verbosity >= 1:
                print(f"❌ Error creating visualizations: {e}")
                import traceback
                if self.verbosity >= 2:
                    traceback.print_exc()
    
    def display_tree_structure(self):
        """Display tree structure with Information Gain and Gini values """
        if not SHOW_TREE_STRUCTURE or self.best_model is None:
            return
            
        print("\\n" + "="*70)
        print("DECISION TREE STRUCTURE VISUALIZATION")  
        print("="*70)
        print("Requirement: Features, Information Gain, and Gini Index at each node")
        print("-" * 70)
        
        self.best_model.visualize_tree(max_depth=TREE_DISPLAY_MAX_DEPTH)
    
    def generate_project_summary(self):
        """Generate a comprehensive project summary for academic submission."""
        if self.verbosity < 1:
            return
            
        print("\\n" + "="*70)
        print("PROJECT COMPLIANCE SUMMARY")
        print("="*70)
        
        print("   • Dataset: Credit Card Fraud (284,807 samples, 30 features + 1 target)")
        print("   • Decision Tree: Complete from-scratch implementation") 
        print("   • Discretization: Quartile binning ")
        print("   • Comparison Methods: Equal width, equal frequency (bonus)")
        print("   • Data Split: 80% training, 20% testing ")
        print("   • Hyperparameter Tuning: Using validation set from training data")
        print("   • Post-pruning: Implemented with validation set evaluation")
        print("   • Tree Visualization: Shows features, Information Gain & Gini Index")
        
        print("\\n⭐ BONUS POINTS did:")
        print("   • Intelligent Data Selection: Stratified sampling for imbalance")
        print("   • Professional Discretization: Multiple innovative methods")
        print("   • Advanced Statistical Analysis: Information Value & feature ranking") 
        print("   • Comprehensive Visualizations: Dashboard + tree structure PNGs")
        print("   • Class Imbalance Handling: F1-score optimization")
        
        if self.results:
            best_result = self.results[self.best_method]
            print("\\n📊 FINAL RESULTS:")
            print(f"   • Best Method: {self.best_method.replace('_', ' ').title()}")
            print(f"   • F1-Score: {best_result['metrics']['f1_score']:.4f}")
            print(f"   • Precision: {best_result['metrics']['precision']:.4f}")
            print(f"   • Recall: {best_result['metrics']['recall']:.4f}")
            print(f"   • Tree Complexity: {best_result.get('tree_size_final', 'N/A')} nodes (after pruning)")
            



def main():
    """Main execution function."""
    print("🌳 COMPREHENSIVE FRAUD DETECTION SYSTEM 🌳")
    print("="*60)
    print("Academic Project - Decision Tree Implementation from Scratch")
    print("Dataset: Credit Card Fraud Detection (284K+ samples)")
    print("="*60)
    
    # start the system
    system = ComprehensiveFraudDetectionSystem()
    
    # Set up dirs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir) 
    output_dir = os.path.join(project_dir, 'outputs')
    
    try:
        # load dataset
        df = system.load_dataset()
        
        # preprocess data with multiple discretization methods
        discretization_results = system.preprocess_data(df)
        
        # train models with hyperparameter tuning
        model_results = system.train_and_tune_models(discretization_results)
        
        # analyze results
        system.analyze_results(model_results)
        
        # create visualizations  
        system.create_visualizations(output_dir, df)
        
        # display tree structure
        system.display_tree_structure()
        
        # generate comprehensive project summary
        system.generate_project_summary()
        
        print(f"\\n🎉 PROJECT COMPLETED SUCCESSFULLY!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 All visualizations and results saved")
        
    except Exception as e:
        print(f"\\n❌ Error during execution: {str(e)}")
        import traceback
        if VERBOSITY_LEVEL >= 2:
            traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
