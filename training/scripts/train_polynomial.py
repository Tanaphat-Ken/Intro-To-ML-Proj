"""
Training script for Polynomial Regression models.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from training.regression.polynomial_regression import PolynomialRegression, MultiVariatePolynomialRegression

# Benchmark configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
POLY_DEGREE = 2
LEARNING_RATE = 0.01
N_ITERATIONS = 1000


def load_data(file_path):
    """Load dataset from CSV file."""
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


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


def main(data_path, degree=POLY_DEGREE, univariate=False):
    """Main training pipeline."""
    print("Loading data...")
    X, y = load_data(data_path)

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

    if univariate and X.shape[1] == 1:
        # Use univariate polynomial regression for single feature
        print("\n" + "="*50)
        print(f"Training Polynomial Regression (degree={degree})...")
        print("="*50)

        poly_model = PolynomialRegression(
            degree=degree,
            learning_rate=LEARNING_RATE,
            n_iterations=N_ITERATIONS
        )
        poly_model.fit(X_train_scaled.flatten(), y_train)

        # Evaluate
        y_train_pred = poly_model.predict(X_train_scaled.flatten())
        y_test_pred = poly_model.predict(X_test_scaled.flatten())

        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        print(f"\nPolynomial Regression (degree={degree}) Results:")
        print(f"{'='*50}")
        print(f"Train MSE: {train_mse:.4f}")
        print(f"Test MSE:  {test_mse:.4f}")
        print(f"Train R²:  {train_r2:.4f}")
        print(f"Test R²:   {test_r2:.4f}")

        # Plot loss curve
        print("\nPlotting loss curve...")
        poly_model.plot_loss()

    else:
        # Use multivariate polynomial regression
        print("\n" + "="*50)
        print(f"Training Multivariate Polynomial Regression (degree={degree})...")
        print("="*50)

        poly_mv_model = MultiVariatePolynomialRegression(degree=degree)
        poly_mv_model.fit(X_train_scaled, y_train)
        poly_mv_results = evaluate_model(poly_mv_model, X_train_scaled, X_test_scaled,
                                        y_train, y_test,
                                        f"Multivariate Polynomial Regression (degree={degree})")

        print(f"\nNumber of polynomial features created: {poly_mv_model._create_polynomial_features(X_train_scaled).shape[1]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train Polynomial Regression models')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--degree', type=int, default=POLY_DEGREE, help='Polynomial degree')
    parser.add_argument('--univariate', action='store_true', help='Use univariate polynomial regression')

    args = parser.parse_args()
    main(args.data, degree=args.degree, univariate=args.univariate)
