"""
Training script for Logistic Regression models.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

from training.classification.logistic_regression import LogisticRegressionMSE, LogisticRegressionBCE

# Benchmark configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
LEARNING_RATE = 0.01
N_ITERATIONS = 1000


def load_data(file_path):
    """Load dataset from CSV file."""
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate classification model and print metrics."""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

    # ROC-AUC if binary classification
    try:
        test_roc_auc = roc_auc_score(y_test, y_test_proba)
    except:
        test_roc_auc = None

    print(f"\n{model_name} Results:")
    print(f"{'='*50}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print(f"Test F1-Score:  {test_f1:.4f}")
    if test_roc_auc is not None:
        print(f"Test ROC-AUC:   {test_roc_auc:.4f}")

    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_roc_auc': test_roc_auc
    }


def main(data_path):
    """Main training pipeline."""
    print("Loading data...")
    X, y = load_data(data_path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Dataset shape: {X.shape}")
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Class distribution: {np.bincount(y.astype(int))}")

    # Train Logistic Regression with MSE loss
    print("\n" + "="*50)
    print("Training Logistic Regression (MSE Loss)...")
    print("="*50)

    lr_mse_model = LogisticRegressionMSE(
        learning_rate=LEARNING_RATE,
        n_iterations=N_ITERATIONS
    )
    lr_mse_model.fit(X_train_scaled, y_train)
    lr_mse_results = evaluate_model(lr_mse_model, X_train_scaled, X_test_scaled,
                                    y_train, y_test, "Logistic Regression (MSE)")

    # Plot loss curve
    print("\nPlotting MSE loss curve...")
    lr_mse_model.plot_loss()

    # Train Logistic Regression with BCE loss
    print("\n" + "="*50)
    print("Training Logistic Regression (BCE Loss)...")
    print("="*50)

    lr_bce_model = LogisticRegressionBCE(
        learning_rate=LEARNING_RATE,
        n_iterations=N_ITERATIONS
    )
    lr_bce_model.fit(X_train_scaled, y_train)
    lr_bce_results = evaluate_model(lr_bce_model, X_train_scaled, X_test_scaled,
                                    y_train, y_test, "Logistic Regression (BCE)")

    # Plot loss curve
    print("\nPlotting BCE loss curve...")
    lr_bce_model.plot_loss()

    # Compare results
    print("\n" + "="*50)
    print("Model Comparison")
    print("="*50)
    print(f"{'Metric':<20} {'MSE Loss':<15} {'BCE Loss':<15}")
    print("-"*50)
    print(f"{'Test Accuracy':<20} {lr_mse_results['test_acc']:<15.4f} {lr_bce_results['test_acc']:<15.4f}")
    print(f"{'Test F1-Score':<20} {lr_mse_results['test_f1']:<15.4f} {lr_bce_results['test_f1']:<15.4f}")
    if lr_mse_results['test_roc_auc'] is not None:
        print(f"{'Test ROC-AUC':<20} {lr_mse_results['test_roc_auc']:<15.4f} {lr_bce_results['test_roc_auc']:<15.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train Logistic Regression models')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')

    args = parser.parse_args()
    main(args.data)
