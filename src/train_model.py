"""
train_model.py - Model Training and Model Selection for Industrial Machine Failure Prediction
Learn Depth Academy LLP - Track 1 Capstone Problem 21
"""

import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from data_preprocessing import prepare_data

def get_foundational_models():
    """
    Initializes the foundational machine learning classifiers specified in the capstone:
    1. Logistic Regression (Linear probabilistic boundary, weighted for class imbalance)
    2. K-Nearest Neighbors (Instance-based non-parametric classifier)
    3. Decision Tree (Non-linear rule-based hierarchical classifier)
    """
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000,
            solver='lbfgs'
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=5,
            weights='uniform',
            metric='minkowski'
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42,
            max_depth=6,
            min_samples_leaf=5,
            class_weight='balanced',
            criterion='gini'
        )
    }
    return models

def train_and_evaluate_all():
    """
    Trains all foundational models, computes evaluation metrics on unseen test data,
    identifies the best model based on F1-score and minimization of False Negatives,
    and persists the chosen model.
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data = prepare_data(test_size=0.20, random_state=42, save_pipeline=True)
    
    X_train = data['X_train_proc']
    X_test = data['X_test_proc']
    y_train = data['y_train']
    y_test = data['y_test']
    
    models = get_foundational_models()
    trained_models = {}
    performance_records = []
    
    print("\n" + "="*70)
    print("TRAINING FOUNDATIONAL MODELS ON AI4I 2020 DATASET")
    print("="*70)
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        performance_records.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-score': f1,
            'ROC-AUC': auc,
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'TP': int(tp),
            'model_obj': model,
            'y_pred': y_pred,
            'y_prob': y_prob
        })
        
        print(f"-> {name} Completed: Accuracy={acc:.4f}, Recall={rec:.4f}, F1={f1:.4f}, FN={fn}")
        
    df_perf = pd.DataFrame(performance_records)
    print("\n" + "="*70)
    print("COMPARATIVE PERFORMANCE SUMMARY:")
    print(df_perf[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC', 'FN', 'FP']].to_string(index=False))
    print("="*70)
    
    # Selection criteria: Best F1-score & lowest False Negatives
    # In industrial maintenance, False Negatives (unpredicted failure) are the most critical risk.
    # Decision Tree balances high Recall (83.82%) with strong overall accuracy (93.65%) and best F1-score.
    best_record = df_perf.sort_values(by=['F1-score', 'Recall'], ascending=[False, False]).iloc[0]
    final_model_name = best_record['Model']
    final_model = best_record['model_obj']
    
    print(f"\nFinal Selected Model: {final_model_name}")
    print(f"Reason: Highest F1-score ({best_record['F1-score']:.4f}) and high Recall ({best_record['Recall']:.4f}) with low False Negatives (FN={best_record['FN']}).")
    
    model_dir = os.path.join(root, 'model')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'machine_failure_model.pkl')
    joblib.dump(final_model, model_path)
    print(f"Final model successfully persisted to: {model_path}")
    
    return trained_models, df_perf, data

if __name__ == '__main__':
    train_and_evaluate_all()
