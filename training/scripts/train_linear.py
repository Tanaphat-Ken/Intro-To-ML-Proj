import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

from training.regression.linear_regression import LinearRegression, LinearRegressionGD

# Benchmark configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
LEARNING_RATE = 0.01
N_ITERATIONS = 1000

# Output paths
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../../output')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')


def load_data(file_path):
    df = pd.read_csv(file_path)

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        raise ValueError(f"Need at least 2 numeric columns. Found {numeric_df.shape[1]}")

    # Use last numeric column as target
    X = numeric_df.iloc[:, :-1].values
    y = numeric_df.iloc[:, -1].values

    print(f"Selected {X.shape[1]} numeric features from {df.shape[1]} total columns")

    return X, y


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
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


def main(data_path):
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

    # Train Linear Regression (Normal Equation)
    print("\n" + "="*50)
    print("Training Linear Regression (Normal Equation)...")
    print("="*50)

    lr_model = LinearRegression(fit_intercept=True)
    lr_model.fit(X_train_scaled, y_train)
    lr_results = evaluate_model(lr_model, X_train_scaled, X_test_scaled,
                                y_train, y_test, "Linear Regression (Normal Equation)")

    # Train Linear Regression (Gradient Descent)
    print("\n" + "="*50)
    print("Training Linear Regression (Gradient Descent)...")
    print("="*50)

    lr_gd_model = LinearRegressionGD(
        learning_rate=LEARNING_RATE,
        n_iterations=N_ITERATIONS
    )
    lr_gd_model.fit(X_train_scaled, y_train)
    lr_gd_results = evaluate_model(lr_gd_model, X_train_scaled, X_test_scaled,
                                   y_train, y_test, "Linear Regression (Gradient Descent)")

    # Save models
    print("\nSaving models...")
    lr_model.save_weights(os.path.join(MODELS_DIR, 'linear_regression.pkl'))
    lr_gd_model.save_weights(os.path.join(MODELS_DIR, 'linear_regression_gd.pkl'))

    # Plot and save loss curve
    print("\nPlotting loss curve...")
    plot_path = os.path.join(PLOTS_DIR, 'linear_regression_gd_loss.png')
    lr_gd_model.plot_loss(save_path=plot_path)

    # Compare results
    print("\n" + "="*50)
    print("Model Comparison")
    print("="*50)
    print(f"{'Metric':<20} {'Normal Eq':<15} {'Gradient Descent':<15}")
    print("-"*50)
    print(f"{'Test MSE':<20} {lr_results['test_mse']:<15.4f} {lr_gd_results['test_mse']:<15.4f}")
    print(f"{'Test R²':<20} {lr_results['test_r2']:<15.4f} {lr_gd_results['test_r2']:<15.4f}")
    print(f"{'Test MAE':<20} {lr_results['test_mae']:<15.4f} {lr_gd_results['test_mae']:<15.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train Linear Regression models')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')

    args = parser.parse_args()
    main(args.data)
