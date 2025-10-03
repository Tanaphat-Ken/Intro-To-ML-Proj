import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from training.regression.regularized_regression import LassoRegression

# Benchmark configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
DEFAULT_ALPHA = 1.0
ALPHA_RANGE = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
LEARNING_RATE = 0.01
N_ITERATIONS = 1000


def load_data(file_path):
    """Load dataset from CSV file."""
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    feature_names = df.columns[:-1].tolist()
    return X, y, feature_names


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate model and print metrics."""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print(f"\n{model_name} Results:")
    print(f"{'='*50}")
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Test MSE:  {test_mse:.4f}")
    print(f"Train R²:  {train_r2:.4f}")
    print(f"Test R²:   {test_r2:.4f}")
    print(f"Test MAE:  {test_mae:.4f}")

    return {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mae': test_mae
    }


def tune_alpha(X_train, X_test, y_train, y_test, alpha_range):
    print("\n" + "="*50)
    print("Hyperparameter Tuning (Alpha)")
    print("="*50)

    best_alpha = None
    best_score = float('inf')
    results = []

    for alpha in alpha_range:
        model = LassoRegression(
            alpha=alpha,
            learning_rate=LEARNING_RATE,
            n_iterations=N_ITERATIONS
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_pred)

        results.append({'alpha': alpha, 'test_mse': test_mse})
        print(f"Alpha: {alpha:>7.3f} | Test MSE: {test_mse:.4f}")

        if test_mse < best_score:
            best_score = test_mse
            best_alpha = alpha

    print(f"\nBest Alpha: {best_alpha} (Test MSE: {best_score:.4f})")
    return best_alpha, results


def main(data_path, tune=False):
    """Main training pipeline."""
    print("Loading data...")
    X, y, feature_names = load_data(data_path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Dataset shape: {X.shape}")
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Tune alpha if requested
    if tune:
        best_alpha, tuning_results = tune_alpha(
            X_train_scaled, X_test_scaled, y_train, y_test, ALPHA_RANGE
        )
    else:
        best_alpha = DEFAULT_ALPHA

    # Train Lasso Regression with best/default alpha
    print("\n" + "="*50)
    print(f"Training Lasso Regression (alpha={best_alpha})...")
    print("="*50)

    lasso_model = LassoRegression(
        alpha=best_alpha,
        learning_rate=LEARNING_RATE,
        n_iterations=N_ITERATIONS
    )
    lasso_model.fit(X_train_scaled, y_train)
    lasso_results = evaluate_model(lasso_model, X_train_scaled, X_test_scaled,
                                   y_train, y_test, f"Lasso Regression (α={best_alpha})")

    # Feature selection
    print("\n" + "="*50)
    print("Feature Selection (L1 Sparsity)")
    print("="*50)
    selected_features = lasso_model.get_selected_features(feature_names, threshold=1e-5)
    print(f"Selected {len(selected_features)} out of {len(feature_names)} features:")
    for feat, coef in selected_features.items():
        print(f"  {feat}: {coef:.4f}")

    # Plot loss curve
    print("\nPlotting loss curve...")
    lasso_model.plot_loss()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train Lasso Regression model')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--tune', action='store_true', help='Tune alpha hyperparameter')

    args = parser.parse_args()
    main(args.data, tune=args.tune)
