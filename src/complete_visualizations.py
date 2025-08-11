"""
========================================================

Complete Visualization Module for Fraud Detection Project
with robust error handling and support for various data formats.

Author: SAMANGHO  
Date: 2025-08-10

========================================================

"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set professional style
plt.style.use('default')
sns.set_palette("husl")


class ComprehensiveVisualizer:
    """
    Professional visualization suite for academic fraud detection project.
    Handles all visualization requirements with robust error handling.
    """
    
    def __init__(self, output_dir: str, dpi: int = 300):
        self.output_dir = output_dir
        self.dpi = dpi
        os.makedirs(output_dir, exist_ok=True)
        
        # Professional color schemes
        self.colors = {
            'primary': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],
            'fraud': '#E74C3C',
            'legitimate': '#27AE60', 
            'neutral': '#95A5A6',
            'accent': '#F39C12'
        }
        
    def create_comprehensive_dashboard(self, results: Dict[str, Any], 
                                     dataset_info: Dict[str, Any] = None) -> str:
        """
        Create a comprehensive dashboard with all key visualizations.
        
        Args:
            results: Dictionary of model results
            dataset_info: Dataset information
            
        Returns:
            Path to saved dashboard
        """
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 16))
            
            # Dataset overview (if available)
            if dataset_info:
                self._plot_dataset_overview(fig, dataset_info)
            
            # Model comparison
            self._plot_model_comparison(fig, results)
            
            # Best model detailed analysis
            best_method = max(results.keys(), 
                            key=lambda k: results[k]['metrics'].get('f1_score', 0) or results[k]['metrics'].get('f1', 0))
            
            self._plot_confusion_matrix(fig, results[best_method])
            self._plot_performance_metrics(fig, results[best_method])
            self._plot_tree_complexity(fig, results)
            self._plot_project_summary(fig, results, best_method)
            
            # Save dashboard
            plt.tight_layout(pad=3.0)
            dashboard_path = os.path.join(self.output_dir, 'comprehensive_dashboard.png')
            plt.savefig(dashboard_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return dashboard_path
            
        except Exception as e:
            print(f"Error creating dashboard: {e}")
            return None
    
    def _plot_dataset_overview(self, fig, dataset_info: Dict[str, Any]):
        """Plot dataset overview information."""
        ax = fig.add_subplot(3, 4, 1)
        
        try:
            # Create dataset summary visualization
            fraud_cases = dataset_info.get('fraud_cases', 0)
            total_samples = dataset_info.get('total_samples', 0)
            legitimate_cases = total_samples - fraud_cases
            
            # Pie chart for class distribution
            sizes = [legitimate_cases, fraud_cases]
            labels = ['Legitimate', 'Fraud']
            colors = [self.colors['legitimate'], self.colors['fraud']]
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                            autopct='%1.2f%%', startangle=90)
            
            ax.set_title('Dataset Class Distribution\\n({:,} total samples)'.format(total_samples), 
                        fontsize=12, fontweight='bold', pad=20)
            
            # Add statistics text
            fraud_rate = (fraud_cases / total_samples * 100) if total_samples > 0 else 0
            ax.text(0, -1.3, f'Challenge: Highly Imbalanced\\nFraud Rate: {fraud_rate:.3f}%', 
                   ha='center', fontsize=10, style='italic')
                   
        except Exception as e:
            ax.text(0.5, 0.5, f'Dataset Overview\\nUnavailable\\n({str(e)})', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
    
    def _plot_model_comparison(self, fig, results: Dict[str, Any]):
        """Plot model performance comparison."""
        ax = fig.add_subplot(3, 4, 2)
        
        try:
            methods = []
            f1_scores = []
            
            for method, result in results.items():
                if 'metrics' in result:
                    methods.append(method.replace('_', '\\n').title())
                    # Handle both 'f1' and 'f1_score' keys
                    f1 = result['metrics'].get('f1_score', result['metrics'].get('f1', 0))
                    f1_scores.append(f1 if f1 is not None and not np.isnan(f1) else 0)
            
            if methods and f1_scores:
                bars = ax.bar(range(len(methods)), f1_scores, 
                            color=self.colors['primary'][:len(methods)])
                
                ax.set_title('Model Performance Comparison\\n(F1-Score)', 
                           fontsize=12, fontweight='bold', pad=20)
                ax.set_ylabel('F1-Score')
                ax.set_xticks(range(len(methods)))
                ax.set_xticklabels(methods, fontsize=9, rotation=45)
                ax.set_ylim(0, 1.1)
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for i, (bar, score) in enumerate(zip(bars, f1_scores)):
                    if score > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                               f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No valid results\\nfor comparison', 
                       ha='center', va='center', transform=ax.transAxes)
                       
        except Exception as e:
            ax.text(0.5, 0.5, f'Model Comparison\\nError: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
    
    def _plot_confusion_matrix(self, fig, best_result: Dict[str, Any]):
        """Plot confusion matrix for best model."""
        ax = fig.add_subplot(3, 4, 3)
        
        try:
            cm = best_result['metrics']['confusion_matrix']
            if isinstance(cm, list):
                cm = np.array(cm)
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Legitimate', 'Fraud'], 
                       yticklabels=['Legitimate', 'Fraud'],
                       cbar_kws={'shrink': 0.8})
            
            ax.set_title('Best Model\\nConfusion Matrix', 
                        fontsize=12, fontweight='bold', pad=20)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Confusion Matrix\\nError: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
    
    def _plot_performance_metrics(self, fig, best_result: Dict[str, Any]):
        """Plot detailed performance metrics."""
        ax = fig.add_subplot(3, 4, 4)
        
        try:
            metrics = best_result['metrics']
            
            # Extract metric values with fallbacks
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            metric_values = [
                metrics.get('accuracy', 0) or 0,
                metrics.get('precision', 0) or 0,
                metrics.get('recall', 0) or 0,
                metrics.get('f1_score', metrics.get('f1', 0)) or 0
            ]
            
            # Handle NaN values
            metric_values = [v if not np.isnan(v) else 0 for v in metric_values]
            
            bars = ax.bar(metric_names, metric_values, 
                         color=self.colors['primary'][:len(metric_names)])
            
            ax.set_title('Best Model\\nPerformance Metrics', 
                        fontsize=12, fontweight='bold', pad=20)
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', alpha=0.3)
            
            # Rotate labels for better fit
            ax.set_xticklabels(metric_names, rotation=45, ha='right')
            
            # Add value labels
            for bar, value in zip(bars, metric_values):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
                           
        except Exception as e:
            ax.text(0.5, 0.5, f'Performance Metrics\\nError: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
    
    def _plot_tree_complexity(self, fig, results: Dict[str, Any]):
        """Plot tree complexity comparison."""
        ax = fig.add_subplot(3, 4, 5)
        
        try:
            methods = []
            initial_sizes = []
            final_sizes = []
            
            for method, result in results.items():
                if 'tree_size_initial' in result and 'tree_size_final' in result:
                    methods.append(method.replace('_', '\\n').title())
                    initial_sizes.append(result['tree_size_initial'])
                    final_sizes.append(result['tree_size_final'])
            
            if methods:
                x = np.arange(len(methods))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, initial_sizes, width, 
                              label='Before Pruning', color=self.colors['fraud'], alpha=0.7)
                bars2 = ax.bar(x + width/2, final_sizes, width,
                              label='After Pruning', color=self.colors['legitimate'], alpha=0.7)
                
                ax.set_title('Tree Complexity\\n(Pruning Effect)', 
                           fontsize=12, fontweight='bold', pad=20)
                ax.set_ylabel('Number of Nodes')
                ax.set_xticks(x)
                ax.set_xticklabels(methods, fontsize=9, rotation=45)
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                               f'{int(height)}', ha='center', va='bottom', fontsize=9)
            else:
                ax.text(0.5, 0.5, 'Tree complexity\\ndata unavailable', 
                       ha='center', va='center', transform=ax.transAxes)
                       
        except Exception as e:
            ax.text(0.5, 0.5, f'Tree Complexity\\nError: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
    
    def _plot_project_summary(self, fig, results: Dict[str, Any], best_method: str):
        """Plot project summary and compliance."""
        ax = fig.add_subplot(3, 4, 6)
        ax.axis('off')
        
        try:
            best_result = results[best_method]
            best_f1 = best_result['metrics'].get('f1_score', best_result['metrics'].get('f1', 0))
            best_f1 = best_f1 if best_f1 is not None and not np.isnan(best_f1) else 0
            
            summary_text = f"""✅ PROJECT COMPLIANCE SUMMARY
            

📊 BEST RESULTS:
• Method: {best_method.replace('_', ' ').title()}
• F1-Score: {best_f1:.4f}
• Tree Size: {best_result.get('tree_size_final', 'N/A')} nodes

🎓 ACADEMIC SUBMISSION READY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"""
            
            ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
                   
        except Exception as e:
            ax.text(0.5, 0.5, f'Project Summary\\nError: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
    
    def create_tree_structure_png(self, model, filename: str = 'decision_tree_structure.png') -> str:
        """
        Create a professional decision tree structure visualization.
        
        Args:
            model: Trained decision tree model
            filename: Output filename
            
        Returns:
            Path to saved image
        """
        try:
            fig, ax = plt.subplots(figsize=(16, 12))
            
            # Create tree visualization using text-based approach with better formatting
            tree_text = self._generate_tree_text(model)
            
            ax.text(0.05, 0.95, tree_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=1", facecolor="white", edgecolor="black"))
            
            ax.set_title('Decision Tree Structure\\nFeatures, Information Gain & Gini Index', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.axis('off')
            # Add legend
            legend_text = """Legend:
🗺 Decision Node: Feature ≤ Threshold
🟢 Leaf Node: Low Fraud Risk  
🔴 Leaf Node: High Fraud Risk
IG: Information Gain
GI: Gini Index"""
            
            ax.text(0.75, 0.25, legend_text, transform=ax.transAxes, 
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
            
            # Save the tree structure
            tree_path = os.path.join(self.output_dir, filename)
            plt.savefig(tree_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return tree_path
            
        except Exception as e:
            print(f"Error creating tree structure: {e}")
            return None
    
    def _generate_tree_text(self, model, max_depth: int = 5) -> str:
        """Generate formatted text representation of the tree."""
        if model.root is None:
            return "Tree not trained"
            
        def _format_node(node, depth=0, side="root", max_depth=max_depth):
            if depth > max_depth or node is None:
                return ""
                
            indent = "  " * depth
            result = ""
            
            if node.prediction is not None:
                # Leaf node
                risk = "HIGH 🔴" if node.prediction == 1 else "LOW 🟢"
                result += f"{indent}{side}: LEAF → FRAUD RISK: {risk}\\n"
                result += f"{indent}    Samples: {node.samples}\\n"
            else:
                # Decision node  
                feature_name = node.feature_name if node.feature_name else f"Feature_{node.feature}"
                result += f"{indent}{side}: 🟦 {feature_name} ≤ {node.threshold:.4f}\\n"
                result += f"{indent}    IG: {node.info_gain:.4f}, GI: {node.gini_index:.4f}\\n"
                result += f"{indent}    Samples: {node.samples}, Depth: {depth}\\n"
                
                if node.left is not None:
                    result += _format_node(node.left, depth + 1, "├─ left", max_depth)
                if node.right is not None:
                    result += _format_node(node.right, depth + 1, "└─ right", max_depth)
                    
            return result
        
        return _format_node(model.root)


def create_all_visualizations(results: Dict[str, Any], 
                            dataset_info: Dict[str, Any],
                            output_dir: str,
                            best_model=None) -> List[str]:
    """
    Create all visualizations for the fraud detection project.
    
    Args:
        results: Model results dictionary
        dataset_info: Dataset information
        output_dir: Output directory path
        best_model: Best performing model instance
        
    Returns:
        List of created file paths
    """
    visualizer = ComprehensiveVisualizer(output_dir)
    created_files = []
    
    try:
        # Create comprehensive dashboard
        dashboard_path = visualizer.create_comprehensive_dashboard(results, dataset_info)
        if dashboard_path:
            created_files.append(dashboard_path)
            print(f"✅ Dashboard created: {dashboard_path}")
        
        # Create tree structure visualization
        if best_model:
            tree_path = visualizer.create_tree_structure_png(best_model)
            if tree_path:
                created_files.append(tree_path)
                print(f"✅ Tree structure created: {tree_path}")
        
        return created_files
        
    except Exception as e:
        print(f"Error in visualization creation: {e}")
        return created_files


# Backward compatibility wrapper functions
def create_visualizations(results: Dict[str, Any], dataset_info: Dict[str, Any], output_dir: str) -> List[str]:
    """
    Wrapper function for backward compatibility with main script.
    """
    # Find best model from results
    best_model = None
    if results:
        best_method = max(results.keys(), 
                         key=lambda k: results[k]['metrics'].get('f1_score', 
                                                               results[k]['metrics'].get('f1', 0)) or 0)
        best_model = results[best_method].get('model')
    
    return create_all_visualizations(results, dataset_info, output_dir, best_model)


def create_tree_structure_visualization(model, output_dir: str) -> str:
    """
    Wrapper function for backward compatibility with main script.
    """
    visualizer = ComprehensiveVisualizer(output_dir)
    return visualizer.create_tree_structure_png(model)
