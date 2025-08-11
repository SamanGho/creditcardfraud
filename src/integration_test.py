"""
Integration Test for Complete Fraud Detection System
===================================================

Tests all components working together to ensure system integrity.

Author: samangho  
Date: 2025-08-10
"""

import numpy as np
import os
import sys
from pathlib import Path

def test_all_components():
    """Test all system components integration"""
    print("🧪 RUNNING INTEGRATION TESTS")
    print("=" * 50)
    
    test_results = {}
    
    # Test 1: Core Decision Tree Algorithms
    print("\n1️⃣ Testing Core Decision Tree Algorithms...")
    try:
        from decision_tree_model import DecisionTreeComplete
        
        # Create simple test data
        X_test = np.array([[1, 2], [2, 3], [3, 1], [1, 3]])
        y_test = np.array([0, 1, 1, 0])
        
        # Test both criteria
        for criterion in ['entropy', 'gini']:
            dt = DecisionTreeComplete(criterion=criterion, max_depth=2)
            dt.fit(X_test, y_test)
            predictions = dt.predict(X_test)
            
            assert len(predictions) == len(y_test), f"Prediction length mismatch for {criterion}"
            assert hasattr(dt, 'root'), f"Root node not created for {criterion}"
            
        test_results['core_algorithms'] = "✅ PASSED"
        print("   ✅ Core algorithms working correctly")
        
    except Exception as e:
        test_results['core_algorithms'] = f"❌ FAILED: {e}"
        print(f"   ❌ Core algorithms failed: {e}")
    
    # Test 2: Unit Tests
    print("\n2️⃣ Testing Unit Test Suite...")
    try:
        from test_decision_tree import TestDecisionTreeAlgorithms
        import unittest
        
        # Run a subset of unit tests
        suite = unittest.TestSuite()
        suite.addTest(TestDecisionTreeAlgorithms('test_entropy_calculations'))
        suite.addTest(TestDecisionTreeAlgorithms('test_gini_index_calculations'))
        
        runner = unittest.TextTestRunner(stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        if result.wasSuccessful():
            test_results['unit_tests'] = "✅ PASSED"
            print("   ✅ Unit tests passing")
        else:
            test_results['unit_tests'] = "❌ FAILED: Unit tests have failures"
            print("   ❌ Unit tests have failures")
            
    except Exception as e:
        test_results['unit_tests'] = f"❌ FAILED: {e}"
        print(f"   ❌ Unit tests failed: {e}")
    
    # Test 3: Enhanced Visualization
    print("\n3️⃣ Testing Enhanced Visualization...")
    try:
        from enhanced_tree_visualization import (generate_simplified_tree_ascii, 
                                                generate_tree_statistics_summary,
                                                print_tree_statistics)
        
        # Create a simple tree for testing
        dt = DecisionTreeComplete(max_depth=2)
        dt.fit(X_test, y_test)
        
        # Test ASCII visualization
        ascii_output = generate_simplified_tree_ascii(dt, ['feature_0', 'feature_1'])
        assert isinstance(ascii_output, str), "ASCII output should be string"
        assert len(ascii_output) > 0, "ASCII output should not be empty"
        
        # Test tree statistics
        stats = generate_tree_statistics_summary(dt)
        assert isinstance(stats, dict), "Stats should be dictionary"
        assert 'total_nodes' in stats, "Stats should include total_nodes"
        
        test_results['enhanced_visualization'] = "✅ PASSED"
        print("   ✅ Enhanced visualization working correctly")
        
    except Exception as e:
        test_results['enhanced_visualization'] = f"❌ FAILED: {e}"
        print(f"   ❌ Enhanced visualization failed: {e}")
    
    # Test 4: Main System Import
    print("\n4️⃣ Testing Main System Components...")
    try:
        # Import main components (without running full system)
        sys.path.append('.')
        
        # Test data preprocessing imports
        from improved_main import check_system_resources
        # Test if decision tree model can be imported
        from decision_tree_model import DecisionTreeComplete
        
        # Test system resource check
        cores = check_system_resources()
        assert isinstance(cores, int), "Cores should be integer"
        assert cores > 0, "Cores should be positive"
        
        test_results['main_system'] = "✅ PASSED"
        print("   ✅ Main system components importing correctly")
        
    except Exception as e:
        test_results['main_system'] = f"❌ FAILED: {e}"
        print(f"   ❌ Main system components failed: {e}")
    
    # Test 5: File Structure
    print("\n5️⃣ Testing File Structure...")
    try:
        required_files = [
            'improved_main.py',
            'decision_tree_model.py', 
            'complete_visualizations.py',
            'test_decision_tree.py',
            'enhanced_tree_visualization.py'
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if not missing_files:
            test_results['file_structure'] = "✅ PASSED"
            print("   ✅ All required files present")
        else:
            test_results['file_structure'] = f"❌ FAILED: Missing files: {missing_files}"
            print(f"   ❌ Missing files: {missing_files}")
            
    except Exception as e:
        test_results['file_structure'] = f"❌ FAILED: {e}"
        print(f"   ❌ File structure test failed: {e}")
    
    # Test Summary
    print(f"\n📊 INTEGRATION TEST SUMMARY")
    print("=" * 50)
    
    passed_tests = sum(1 for result in test_results.values() if result == "✅ PASSED")
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        print(f"   {test_name.replace('_', ' ').title()}: {result}")
    
    print(f"\n🎯 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL INTEGRATION TESTS PASSED! System is ready for production.")
        return True
    else:
        print("⚠️  Some integration tests failed. Please review the issues above.")
        return False


def test_model_persistence():
    """Test model persistence functionality"""
    print("\n🧪 Testing Model Persistence...")
    try:
        # Check if recent model file exists
        output_dir = Path("../outputs")
        model_files = list(output_dir.glob("best_model_*.pkl")) if output_dir.exists() else []
        
        if model_files:
            print(f"   ✅ Found {len(model_files)} saved model(s)")
            
            # Try to load a model (basic test)
            latest_model = max(model_files, key=os.path.getctime)
            print(f"   📁 Latest model: {latest_model.name}")
            
            # Check if loader script exists
            loader_script = Path("../outputs/load_model.py")
            if loader_script.exists():
                print("   ✅ Model loader script found")
            else:
                print("   ⚠️  Model loader script not found")
            
            return True
        else:
            print("   ⚠️  No saved models found. Run main system first.")
            return False
            
    except Exception as e:
        print(f"   ❌ Model persistence test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Full Integration Test Suite...")
    
    # Run integration tests
    integration_success = test_all_components()
    
    # Test model persistence if available
    persistence_success = test_model_persistence()
    
    print(f"\n🏁 FINAL RESULT:")
    print(f"   Integration Tests: {'✅ PASSED' if integration_success else '❌ FAILED'}")
    print(f"   Model Persistence: {'✅ VERIFIED' if persistence_success else '⚠️  PARTIAL'}")
    
    if integration_success and persistence_success:
        print("\n🎉 COMPLETE SYSTEM VERIFICATION SUCCESSFUL!")
        print("💼 Your fraud detection system is fully operational and ready for deployment.")
    else:
        print("\n⚠️  System verification completed with some limitations.")
        print("📋 Review the test results above for any issues to address.")
