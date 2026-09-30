"""
data_preprocessing.py - Data Ingestion, Quality Audit, Leakage Prevention & Pipeline Construction
Learn Depth Academy LLP - Track 1 Capstone Problem 21
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_data(filepath=None):
    """Loads the AI4I 2020 dataset from the dataset directory."""
    if filepath is None:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        filepath = os.path.join(root, 'dataset', 'ai4i2020.csv')
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    df = pd.read_csv(filepath)
    return df

def audit_data_quality(df):
    """
    Performs rigorous data quality checks:
    - Shape, data types, missing values, duplicates, constant columns, class balance.
    """
    audit = {}
    audit['shape'] = df.shape
    audit['missing_values'] = df.isnull().sum().to_dict()
    audit['total_missing'] = int(df.isnull().sum().sum())
    audit['duplicates'] = int(df.duplicated().sum())
    audit['dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    # Check for constant columns
    audit['constant_columns'] = [col for col in df.columns if df[col].nunique() <= 1]
    
    # Target variable analysis
    if 'Machine failure' in df.columns:
        counts = df['Machine failure'].value_counts().to_dict()
        percentages = (df['Machine failure'].value_counts(normalize=True) * 100).to_dict()
        audit['target_counts'] = counts
        audit['target_percentages'] = percentages
        audit['imbalance_ratio'] = round(counts.get(0, 0) / max(counts.get(1, 1), 1), 2)
        
    return audit

def prevent_data_leakage(df):
    """
    Explicitly segregates features to prevent data leakage:
    - Identifiers: UDI, Product ID (excluded)
    - Failure Modes: TWF, HDF, PWF, OSF, RNF (excluded as they are post-failure flags)
    - Target: Machine failure
    - Predictive Features: Type, Air temperature [K], Process temperature [K],
                           Rotational speed [rpm], Torque [Nm], Tool wear [min]
    """
    leakage_cols = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    target_col = 'Machine failure'
    
    feature_cols = [
        'Type',
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    return X, y, feature_cols, leakage_cols

def build_preprocessing_pipeline():
    """
    Constructs a ColumnTransformer pipeline:
    - Numerical features: StandardScaler()
    - Categorical features: OneHotEncoder(drop='first', sparse_output=False)
    """
    num_features = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]
    cat_features = ['Type']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_features)
        ],
        remainder='drop'
    )
    return preprocessor

def prepare_data(test_size=0.20, random_state=42, save_pipeline=True):
    """
    Full pipeline execution:
    1. Load data
    2. Audit quality
    3. Separate features and target (preventing leakage)
    4. Stratified Train-Test split
    5. Fit ColumnTransformer ONLY on training split
    6. Transform test split
    7. Save fitted pipeline
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    df = load_data()
    
    audit = audit_data_quality(df)
    X, y, feature_cols, leakage_cols = prevent_data_leakage(df)
    
    # Stratified Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    preprocessor = build_preprocessing_pipeline()
    
    # Fit ONLY on training data
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    if save_pipeline:
        model_dir = os.path.join(root, 'model')
        os.makedirs(model_dir, exist_ok=True)
        pipeline_path = os.path.join(model_dir, 'preprocessing_pipeline.pkl')
        joblib.dump(preprocessor, pipeline_path)
        print(f"Preprocessing pipeline fitted and saved successfully to: {pipeline_path}")
        
    return {
        'X_train_raw': X_train,
        'X_test_raw': X_test,
        'X_train_proc': X_train_proc,
        'X_test_proc': X_test_proc,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'feature_cols': feature_cols,
        'leakage_cols': leakage_cols,
        'audit': audit
    }

if __name__ == '__main__':
    data_dict = prepare_data()
    print("Data preparation complete.")
    print("Training set shape:", data_dict['X_train_proc'].shape)
    print("Testing set shape:", data_dict['X_test_proc'].shape)
    print("Class distribution in Train:\n", data_dict['y_train'].value_counts())
    print("Class distribution in Test:\n", data_dict['y_test'].value_counts())
