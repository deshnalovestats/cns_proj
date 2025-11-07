"""
Visualization and Evaluation Tools
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report
)
import json

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

class DetectionVisualizer:
    """Visualize detection results and performance"""
    
    def __init__(self, output_dir='outputs/visualizations'):
        self.output_dir = output_dir
        
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'])
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
        
    def plot_roc_curve(self, y_true, y_scores, save_path=None):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        plt.show()
        
    def plot_precision_recall_curve(self, y_true, y_scores, save_path=None):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PR curve saved to {save_path}")
        
        plt.show()
        
    def plot_model_comparison(self, metrics_dict, save_path=None):
        """Compare performance of different models"""
        models = list(metrics_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            values = [metrics_dict[model][metric] for model in models]
            
            axes[idx].bar(models, values, color=sns.color_palette("husl", len(models)))
            axes[idx].set_title(metric.replace('_', ' ').title(), 
                               fontsize=14, fontweight='bold')
            axes[idx].set_ylabel('Score', fontsize=11)
            axes[idx].set_ylim([0, 1])
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.02, f'{v:.3f}', 
                              ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison saved to {save_path}")
        
        plt.show()
        
    def plot_attack_distribution(self, results_df, save_path=None):
        """Plot distribution of detected attacks"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Attack type distribution
        if 'attack_type' in results_df.columns:
            attack_counts = results_df[results_df['prediction'] == 1]['attack_type'].value_counts()
            axes[0].pie(attack_counts.values, labels=attack_counts.index, autopct='%1.1f%%',
                       colors=sns.color_palette("Set2"))
            axes[0].set_title('Attack Type Distribution', fontsize=14, fontweight='bold')
        
        # Risk level distribution
        if 'risk_level' in results_df.columns:
            risk_counts = results_df[results_df['prediction'] == 1]['risk_level'].value_counts()
            colors = {'LOW': 'green', 'MEDIUM': 'yellow', 'HIGH': 'orange', 'CRITICAL': 'red'}
            risk_colors = [colors.get(level, 'gray') for level in risk_counts.index]
            axes[1].bar(risk_counts.index, risk_counts.values, color=risk_colors)
            axes[1].set_title('Risk Level Distribution', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Count', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attack distribution saved to {save_path}")
        
        plt.show()
        
    def plot_feature_importance(self, feature_importance_df, top_n=20, save_path=None):
        """Plot feature importance"""
        top_features = feature_importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'], 
                color=sns.color_palette("viridis", len(top_features)))
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Top {top_n} Most Important Features', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance saved to {save_path}")
        
        plt.show()
        
    def plot_timeline(self, results_df, save_path=None):
        """Plot attack detection timeline"""
        if 'timestamp' not in results_df.columns:
            print("Timestamp column not found")
            return
        
        results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
        results_df = results_df.sort_values('timestamp')
        
        # Group by hour
        results_df['hour'] = results_df['timestamp'].dt.floor('H')
        timeline = results_df.groupby(['hour', 'prediction']).size().unstack(fill_value=0)
        
        plt.figure(figsize=(14, 6))
        if 0 in timeline.columns:
            plt.plot(timeline.index, timeline[0], label='Normal', color='green', linewidth=2)
        if 1 in timeline.columns:
            plt.plot(timeline.index, timeline[1], label='Attack', color='red', linewidth=2)
        
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Number of Events', fontsize=12)
        plt.title('Attack Detection Timeline', fontsize=16, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Timeline saved to {save_path}")
        
        plt.show()
        
    def plot_score_distribution(self, results_df, save_path=None):
        """Plot distribution of detection scores"""
        if 'score' not in results_df.columns:
            if 'ensemble_score' in results_df.columns:
                score_col = 'ensemble_score'
            else:
                print("Score column not found")
                return
        else:
            score_col = 'score'
        
        plt.figure(figsize=(10, 6))
        
        # Separate normal and attack
        normal_scores = results_df[results_df['is_attack'] == 0][score_col]
        attack_scores = results_df[results_df['is_attack'] == 1][score_col]
        
        plt.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='green', edgecolor='black')
        plt.hist(attack_scores, bins=50, alpha=0.6, label='Attack', color='red', edgecolor='black')
        
        plt.xlabel('Detection Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Detection Score Distribution', fontsize=16, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Score distribution saved to {save_path}")
        
        plt.show()
        
    def generate_all_plots(self, results_df, metrics_path=None):
        """Generate all visualization plots"""
        print("Generating visualizations...")
        
        # Ensure output directory exists
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        y_true = results_df['is_attack']
        y_pred = results_df['prediction']
        
        # Get scores
        if 'score' in results_df.columns:
            y_scores = results_df['score']
        elif 'ensemble_score' in results_df.columns:
            y_scores = results_df['ensemble_score']
        else:
            y_scores = y_pred.astype(float)
        
        # 1. Confusion Matrix
        self.plot_confusion_matrix(y_true, y_pred, 
                                   f'{self.output_dir}/confusion_matrix.png')
        
        # 2. ROC Curve
        if len(np.unique(y_true)) > 1:
            self.plot_roc_curve(y_true, y_scores,
                               f'{self.output_dir}/roc_curve.png')
            
            # 3. PR Curve
            self.plot_precision_recall_curve(y_true, y_scores,
                                            f'{self.output_dir}/pr_curve.png')
        
        # 4. Attack Distribution
        self.plot_attack_distribution(results_df,
                                     f'{self.output_dir}/attack_distribution.png')
        
        # 5. Score Distribution
        self.plot_score_distribution(results_df,
                                    f'{self.output_dir}/score_distribution.png')
        
        # 6. Timeline
        if 'timestamp' in results_df.columns:
            self.plot_timeline(results_df,
                             f'{self.output_dir}/timeline.png')
        
        # 7. Model Comparison (if metrics available)
        if metrics_path and os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            if 'individual_models' in metrics:
                self.plot_model_comparison(metrics['individual_models'],
                                          f'{self.output_dir}/model_comparison.png')
        
        # 8. Feature Importance
        feature_importance_path = 'outputs/reports/feature_importance.csv'
        if os.path.exists(feature_importance_path):
            fi_df = pd.read_csv(feature_importance_path)
            self.plot_feature_importance(fi_df,
                                        f'{self.output_dir}/feature_importance.png')
        
        print(f"\nAll visualizations saved to {self.output_dir}/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate visualizations')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to detection results CSV')
    parser.add_argument('--metrics', type=str, default=None,
                       help='Path to evaluation metrics JSON')
    parser.add_argument('--output', type=str, default='outputs/visualizations',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results}...")
    results_df = pd.read_csv(args.results)
    
    # Create visualizer
    visualizer = DetectionVisualizer(args.output)
    
    # Generate all plots
    visualizer.generate_all_plots(results_df, args.metrics)
    
    print("\nVisualization complete!")
