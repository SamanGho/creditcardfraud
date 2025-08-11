"""
Example Usage of Credit Card Fraud Detection System

This script demonstrates how to use the fraud detection system
with custom parameters and generate specific outputs.
"""

import sys
import os
sys.path.append('src')

from fraud_detection_main import ComprehensiveFraudDetectionSystem
from generate_all_pngs import (
    generate_1_dataset_distribution,
    generate_2_model_performance_comparison
)
from results_extractor import load_real_results

def simple_example():
    """Basic usage example - train and evaluate"""
    print("="*60)
    print("SIMPLE EXAMPLE: Train and Evaluate")
    print("="*60)
    
    # Initialize system
    system = ComprehensiveFraudDetectionSystem()
    
    # Load dataset
    print("\n1. Loading dataset...")
    df = system.load_dataset()
    
    # Preprocess and discretize
    print("\n2. Preprocessing data...")
    discretization_results = system.preprocess_data(df)
    
    # Train models
    print("\n3. Training models...")
    model_results = system.train_and_tune_models(discretization_results)
    
    # Analyze results
    print("\n4. Analyzing results...")
    system.analyze_results(model_results)
    
    print("\n✅ Training complete! Check outputs/ folder for results.")

def custom_config_example():
    """Example with custom configuration"""
    print("\n" + "="*60)
    print("CUSTOM CONFIGURATION EXAMPLE")
    print("="*60)
    
    # Temporarily modify configuration
    import config
    
    # Save original values
    original_sample = config.SAMPLE_SIZE
    original_full = config.USE_FULL_DATASET
    
    # Use smaller sample for faster testing
    config.USE_FULL_DATASET = False
    config.SAMPLE_SIZE = 10000
    
    print(f"Using sample size: {config.SAMPLE_SIZE}")
    
    # Run with custom config
    system = ComprehensiveFraudDetectionSystem()
    df = system.load_dataset()
    
    # Restore original values
    config.USE_FULL_DATASET = original_full
    config.SAMPLE_SIZE = original_sample
    
    print("✅ Custom configuration example complete!")

def visualization_example():
    """Example of generating specific visualizations"""
    print("\n" + "="*60)
    print("VISUALIZATION EXAMPLE")
    print("="*60)
    
    print("\nGenerating specific visualizations...")
    
    try:
        # Generate dataset distribution
        print("1. Creating dataset distribution chart...")
        generate_1_dataset_distribution()
        
        # Generate performance comparison
        print("2. Creating performance comparison...")
        generate_2_model_performance_comparison()
        
        print("\n✅ Visualizations saved to outputs/ folder!")
        
    except Exception as e:
        print(f"⚠️ Make sure models are trained first: {e}")
        print("Run simple_example() first to train models.")

def results_analysis_example():
    """Example of analyzing saved results"""
    print("\n" + "="*60)
    print("RESULTS ANALYSIS EXAMPLE")
    print("="*60)
    
    try:
        # Load saved results
        performance, confusion_matrices, dataset_info = load_real_results()
        
        if performance:
            print("\n📊 Performance Summary:")
            print("-" * 40)
            
            for i, method in enumerate(performance['method']):
                print(f"\n{method} Method:")
                print(f"  F1-Score:  {performance['f1_score'][i]:.4f}")
                print(f"  Precision: {performance['precision'][i]:.4f}")
                print(f"  Recall:    {performance['recall'][i]:.4f}")
                print(f"  Accuracy:  {performance['accuracy'][i]:.4f}")
            
            # Find best model
            best_idx = performance['f1_score'].index(max(performance['f1_score']))
            best_method = performance['method'][best_idx]
            best_f1 = performance['f1_score'][best_idx]
            
            print(f"\n🏆 Best Model: {best_method} (F1-Score: {best_f1:.4f})")
            
            # Dataset info
            print(f"\n📊 Dataset Statistics:")
            print(f"  Total Samples: {dataset_info['total_samples']:,}")
            print(f"  Fraud Cases: {dataset_info['fraud_samples']:,}")
            print(f"  Fraud Rate: {dataset_info['fraud_percentage']:.3f}%")
            
        else:
            print("⚠️ No results found. Train models first using simple_example()")
            
    except Exception as e:
        print(f"Error loading results: {e}")

def main():
    """Main example runner"""
    print("🌳 CREDIT CARD FRAUD DETECTION - EXAMPLES 🌳")
    print("=" * 60)
    print("\nChoose an example to run:")
    print("1. Simple Example (Train and Evaluate)")
    print("2. Custom Configuration Example")
    print("3. Visualization Example")
    print("4. Results Analysis Example")
    print("5. Run All Examples")
    print("0. Exit")
    
    choice = input("\nEnter your choice (0-5): ").strip()
    
    if choice == '1':
        simple_example()
    elif choice == '2':
        custom_config_example()
    elif choice == '3':
        visualization_example()
    elif choice == '4':
        results_analysis_example()
    elif choice == '5':
        simple_example()
        custom_config_example()
        visualization_example()
        results_analysis_example()
    elif choice == '0':
        print("Exiting...")
    else:
        print("Invalid choice. Please run again.")

if __name__ == "__main__":
    # Check if dataset exists
    dataset_path = '../creditcard.csv'  # Dataset is in parent directory
    if not os.path.exists(dataset_path):
        print("⚠️ WARNING: creditcard.csv not found! 😨(just for you ;fellow TA)")
        print("Please download from: https://www.kaggle.com/mlg-ulb/creditcardfraud i guess you gotta open that browser of yours 😞")
        print("and place in the csv in  root directory ")
        sys.exit(1)
    
    main()
