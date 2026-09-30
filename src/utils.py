"""
utils.py - Utility Functions for Industrial Machine Failure Prediction
Learn Depth Academy LLP - Track 1 Capstone Problem 21
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns

def get_project_root():
    """Returns absolute path to project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def setup_plot_style():
    """Sets consistent, publication-quality styling for matplotlib and seaborn."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })

def ensure_dirs():
    """Ensures all standard project directories exist."""
    root = get_project_root()
    dirs = ['dataset', 'notebook', 'src', 'model', 'results', os.path.join('screenshots', 'streamlit'), 'report', 'presentation']
    for d in dirs:
        os.makedirs(os.path.join(root, d), exist_ok=True)
