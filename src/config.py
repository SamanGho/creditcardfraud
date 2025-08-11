"""
Configuration settings for the fraud detection project.
well i gotta explain alot so read the comment and set it up for yourself plz
"""

# Dataset Configuration - Robust defaults for comprehensive analysis
USE_FULL_DATASET = True   # Production setting: Use full dataset for maximum accuracy
SAMPLE_SIZE = 100000      # Large sample for robust analysis when full dataset not used
MIN_SAMPLES_FOR_FULL = 50000  # Minimum samples to consider using full dataset
MAX_TUNING_TIME_MINUTES = 45  # Extended time for thorough hyperparameter search
MIN_FRAUD_CASES_VAL = 15  # Higher minimum for more reliable validation metrics

# Data Splitting Configuration (requires 80/20 train/test)
TRAIN_SIZE = 0.64    # 64% for training (80% of 80%)
VAL_SIZE = 0.16      # 16% for validation (20% of 80%)  
TEST_SIZE = 0.20     # 20% for testing 

# Model Configuration
HYPERPARAMETER_TUNING = True
# Comprehensive parameter grid for thorough analysis
MAX_DEPTH_OPTIONS = [4, 6, 8, 10, 12, None]  # Extended depth range for better coverage
MIN_SAMPLES_SPLIT_OPTIONS = [2, 5, 10, 20, 50]  # More granular split options
MIN_SAMPLES_LEAF_OPTIONS = [1, 2, 5, 10]   # Extended leaf sizes for fraud detection

# Discretization Configuration
PRIMARY_METHOD = 'quartile' 
COMPARISON_METHODS = ['equal_width', 'equal_frequency']

# Output Configuration
VERBOSITY_LEVEL = 1  # 0=silent, 1=essential, 2=detailed, 3=debug
SHOW_TREE_STRUCTURE = True
TREE_DISPLAY_MAX_DEPTH = 4
SAVE_VISUALIZATIONS = True

# Advanced Analysis Features (for bonus points)
ENABLE_ADVANCED_STATS = True
ENABLE_FEATURE_IMPORTANCE = True
ENABLE_ROC_ANALYSIS = True
HANDLE_CLASS_IMBALANCE = True

# Visualization Configuration
CREATE_COMPREHENSIVE_DASHBOARD = True
FIGURE_DPI = 300
FIGURE_SIZE = (12, 8)
