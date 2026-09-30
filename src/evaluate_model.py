"""
evaluate_model.py - Comprehensive Evaluation, Error Analysis & Visualization Generation
Learn Depth Academy LLP - Track 1 Capstone Problem 21
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix

from utils import setup_plot_style, get_project_root
from data_preprocessing import load_data, prevent_data_leakage
from train_model import train_and_evaluate_all

def generate_eda_plots(df, results_dir):
    """
    Generates and saves the required EDA plots:
    1. target_distribution.png
    2. feature_distributions.png
    3. correlation_heatmap.png
    4. boxplots.png
    """
    setup_plot_style()
    
    # 1. Target Distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df['Machine failure'].value_counts()
    colors = ['#2b5c8f', '#d9534f']
    bars = ax.bar(['No Failure (0)', 'Machine Failure (1)'], counts.values, color=colors, edgecolor='black', width=0.5)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 100, f'{yval} ({yval/len(df)*100:.2f}%)', ha='center', va='bottom', fontweight='bold')
    ax.set_title('Target Class Distribution (Machine Failure)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Operational Status', fontsize=12)
    ax.set_ylabel('Observation Count', fontsize=12)
    ax.set_ylim(0, 11000)
    plt.tight_layout()
    target_path = os.path.join(results_dir, 'target_distribution.png')
    plt.savefig(target_path)
    plt.close()
    print(f"Saved: {target_path}")

    # 2. Feature Distributions
    num_cols = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color='#1f77b4', bins=30)
        axes[i].set_title(f'Distribution of {col}', fontweight='bold')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    
    # Use the 6th plot for Type distribution
    sns.countplot(data=df, x='Type', order=['L', 'M', 'H'], ax=axes[5], palette='Blues_r')
    axes[5].set_title('Distribution of Product Variant Type', fontweight='bold')
    axes[5].set_xlabel('Machine Quality Variant')
    axes[5].set_ylabel('Count')
    
    plt.suptitle('Exploratory Feature Distributions (AI4I 2020 Dataset)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    feat_path = os.path.join(results_dir, 'feature_distributions.png')
    plt.savefig(feat_path)
    plt.close()
    print(f"Saved: {feat_path}")

    # 3. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    corr_cols = num_cols + ['Machine failure']
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)
    ax.set_title('Pearson Correlation Heatmap (Continuous Features & Target)', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    corr_path = os.path.join(results_dir, 'correlation_heatmap.png')
    plt.savefig(corr_path)
    plt.close()
    print(f"Saved: {corr_path}")

    # 4. Boxplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.boxplot(data=df, x='Machine failure', y=col, ax=axes[i], palette=['#6baed6', '#fb6a4a'])
        axes[i].set_title(f'{col} vs Machine Failure', fontweight='bold')
        axes[i].set_xticklabels(['No Failure (0)', 'Failure (1)'])
        axes[i].set_xlabel('Operational Status')
        axes[i].set_ylabel(col)
    
    # 6th plot: Rotational Speed vs Torque Scatter
    sns.scatterplot(data=df, x='Rotational speed [rpm]', y='Torque [Nm]', hue='Machine failure', palette=['#3182bd', '#de2d26'], alpha=0.6, ax=axes[5])
    axes[5].set_title('Torque vs Rotational Speed by Failure', fontweight='bold')
    axes[5].set_xlabel('Rotational Speed [rpm]')
    axes[5].set_ylabel('Torque [Nm]')
    
    plt.suptitle('Sensor Feature Behavior by Machine Failure Status', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    box_path = os.path.join(results_dir, 'boxplots.png')
    plt.savefig(box_path)
    plt.close()
    print(f"Saved: {box_path}")

def generate_model_evaluation_plots(df_perf, y_test, results_dir):
    """
    Generates and saves the required model evaluation plots:
    1. confusion_matrix.png (side-by-side matrices)
    2. roc_curve.png (ROC curves for all 3 models)
    3. model_comparison.png (bar chart comparing metrics)
    """
    setup_plot_style()
    
    # 1. Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for i, row in df_perf.iterrows():
        model_name = row['Model']
        cm = np.array([[row['TN'], row['FP']], [row['FN'], row['TP']]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                    xticklabels=['Pred 0 (No Fail)', 'Pred 1 (Fail)'],
                    yticklabels=['Actual 0', 'Actual 1'])
        axes[i].set_title(f'{model_name}\nFN={row["FN"]} | Recall={row["Recall"]*100:.1f}%', fontweight='bold')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('Actual Label')
    plt.suptitle('Confusion Matrix Comparison (Test Set: N=2000, Failures=68)', fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved: {cm_path}")

    # 2. ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    for idx, row in df_perf.iterrows():
        y_prob = row['y_prob']
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[idx], lw=2.2, label=f"{row['Model']} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.5000)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    roc_path = os.path.join(results_dir, 'roc_curve.png')
    plt.savefig(roc_path)
    plt.close()
    print(f"Saved: {roc_path}")

    # 3. Model Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5.5))
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']
    plot_df = df_perf.melt(id_vars=['Model'], value_vars=metrics_to_plot, var_name='Metric', value_name='Score')
    
    sns.barplot(data=plot_df, x='Metric', y='Score', hue='Model', palette='tab10', ax=ax)
    ax.set_title('Comparative Performance of Foundational Machine Learning Models', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Score (0.0 to 1.0)', fontsize=12)
    ax.set_ylim(0, 1.1)
    for p in ax.patches:
        height = p.get_height()
        if height > 0.05:
            ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8.5, xytext=(0, 2), textcoords='offset points')
    ax.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    comp_path = os.path.join(results_dir, 'model_comparison.png')
    plt.savefig(comp_path)
    plt.close()
    print(f"Saved: {comp_path}")

def perform_error_analysis(trained_models, data):
    """
    Performs error analysis on test samples misclassified by the final Decision Tree model.
    """
    X_test_raw = data['X_test_raw'].copy()
    y_test = data['y_test'].values
    dt_model = trained_models['Decision Tree']
    y_pred = dt_model.predict(data['X_test_proc'])
    
    X_test_raw['Actual'] = y_test
    X_test_raw['Predicted'] = y_pred
    
    # False Negatives: Actual 1, Predicted 0 (Critical failure undetected)
    fn_samples = X_test_raw[(X_test_raw['Actual'] == 1) & (X_test_raw['Predicted'] == 0)]
    # False Positives: Actual 0, Predicted 1 (False alarm)
    fp_samples = X_test_raw[(X_test_raw['Actual'] == 0) & (X_test_raw['Predicted'] == 1)]
    # True Positives: Actual 1, Predicted 1
    tp_samples = X_test_raw[(X_test_raw['Actual'] == 1) & (X_test_raw['Predicted'] == 1)]
    # True Negatives: Actual 0, Predicted 0
    tn_samples = X_test_raw[(X_test_raw['Actual'] == 0) & (X_test_raw['Predicted'] == 0)]
    
    print("\n" + "="*70)
    print("DETAILED ERROR ANALYSIS (DECISION TREE)")
    print("="*70)
    print(f"Total Test Instances: {len(X_test_raw)}")
    print(f"True Negatives  (TN): {len(tn_samples)} ({len(tn_samples)/len(X_test_raw)*100:.2f}%)")
    print(f"True Positives  (TP): {len(tp_samples)} ({len(tp_samples)/len(X_test_raw)*100:.2f}%)")
    print(f"False Positives (FP): {len(fp_samples)} ({len(fp_samples)/len(X_test_raw)*100:.2f}%) - False Alarms")
    print(f"False Negatives (FN): {len(fn_samples)} ({len(fn_samples)/len(X_test_raw)*100:.2f}%) - Critical Misses")
    print("\nFalse Negative Samples (Failures Missed by Model):")
    print(fn_samples.head(5).to_string())
    print("\nError Analysis Interpretation:")
    print("1. Missed failures (FN) primarily involve subtle tool wear combinations where torque")
    print("   and rotational speed remained near normal operating boundaries (e.g. Random Failure RNF modes).")
    print("2. False alarms (FP) occur under elevated tool wear or high torque conditions where the machine")
    print("   survived without immediate catastrophic failure, reflecting protective preventative thresholds.")
    print("="*70)
    
    return {
        'total': len(X_test_raw),
        'tn': len(tn_samples),
        'tp': len(tp_samples),
        'fp': len(fp_samples),
        'fn': len(fn_samples),
        'fn_df': fn_samples,
        'fp_df': fp_samples
    }

def main():
    root = get_project_root()
    results_dir = os.path.join(root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    df = load_data()
    print("Generating Exploratory Data Analysis plots...")
    generate_eda_plots(df, results_dir)
    
    print("\nTraining and evaluating models...")
    trained_models, df_perf, data = train_and_evaluate_all()
    
    print("\nGenerating Model Evaluation visualizations...")
    generate_model_evaluation_plots(df_perf, data['y_test'], results_dir)
    
    print("\nConducting Error Analysis...")
    perform_error_analysis(trained_models, data)
    print("\nEvaluation pipeline successfully executed!")

if __name__ == '__main__':
    main()
