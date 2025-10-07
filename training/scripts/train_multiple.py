import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from training.regression.multiple_regression import MultipleLinearRegression, MultipleLinearRegressionGD

# Benchmark configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
LEARNING_RATE = 0.01
N_ITERATIONS = 1000

# Paths (resolve relative to project data dir)
SCRIPT_PATH = Path(__file__).resolve()
SYS_PATH_ROOT = SCRIPT_PATH.parents[2]
DATA_DIR = SYS_PATH_ROOT / "data"
OUTPUT_DIR = SYS_PATH_ROOT / "output"
MODELS_DIR = OUTPUT_DIR / "models"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Ensure output directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_dataset_path(path_str: str) -> Path:
    candidate = Path(path_str)
    potential_paths = [candidate]

    if not candidate.is_absolute():
        potential_paths.extend([Path.cwd() / candidate, DATA_DIR / candidate])

    if candidate.suffix == "":
        potential_paths.extend(
            [p.with_suffix(ext) for p in potential_paths for ext in (".parquet", ".csv")]
        )

    for path in potential_paths:
        if path.exists():
            return path

    searched = "\n".join(str(p) for p in potential_paths)
    raise FileNotFoundError(f"Could not locate dataset '{path_str}'. Paths tried:\n{searched}")


def load_data(file_path, target_column: str | None = None):
    """Load data from CSV or Parquet. If target_column is provided, use it as y. Otherwise
    fall back to previous behaviour: take all columns except last as X, last as y.
    Returns (X, y, feature_names) as numpy arrays.
    """
    dataset_path = _resolve_dataset_path(file_path)
    if dataset_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(dataset_path)
    elif dataset_path.suffix.lower() == ".csv":
        df = pd.read_csv(dataset_path)
    else:
        raise ValueError(f"Unsupported file extension '{dataset_path.suffix}'. Use CSV or Parquet files.")

    # Special case: if this looks like the demand dataset, use demand_count as target
    if 'demand_count' in df.columns:
        if 'datetime_hour' in df.columns:
            feature_df = df.drop(columns=['demand_count', 'datetime_hour'])
        else:
            feature_df = df.drop(columns=['demand_count'])
        y = df['demand_count'].to_numpy()
        # Keep only numeric features for X
        numeric_df = feature_df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] == 0:
            raise ValueError("After dropping 'demand_count' and 'datetime_hour' no numeric features remain")
        X = numeric_df.to_numpy()
        feature_names = numeric_df.columns.tolist()
        print(f"Detected 'demand_count' target. Using {X.shape[1]} numeric features (dropped datetime_hour if present)")
        return X, y, feature_names

    # If explicit target specified, validate and extract
    if target_column:
        if target_column not in df.columns:
            raise KeyError(f"Target column '{target_column}' not found. Available columns: {df.columns.tolist()}")
        y = df[target_column].to_numpy()
        if not np.issubdtype(y.dtype, np.number):
            raise ValueError(f"Target column '{target_column}' must be numeric for regression. Got dtype {df[target_column].dtype}")
        feature_df = df.drop(columns=[target_column])
        # Keep only numeric features
        numeric_df = feature_df.select_dtypes(include=[np.number])
        X = numeric_df.to_numpy()
        feature_names = numeric_df.columns.tolist()
        print(f"Using target column '{target_column}' and {X.shape[1]} numeric features")
        return X, y, feature_names

    # Keep previous behaviour: all columns except last as X, last as y
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    feature_names = df.columns[:-1].tolist()
    print(f"Using {X.shape[1]} features and last column as target")
    return X, y, feature_names


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


def main(data_path, target_column=None):
    """Main training pipeline."""
    print("Loading data...")
    X, y, feature_names = load_data(data_path, target_column)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Train Multiple Linear Regression (Normal Equation)
    print("\n" + "="*50)
    print("Training Multiple Linear Regression (Normal Equation)...")
    print("="*50)

    mlr_model = MultipleLinearRegression(fit_intercept=True)
    mlr_model.fit(X_train_scaled, y_train)
    mlr_results = evaluate_model(mlr_model, X_train_scaled, X_test_scaled,
                                 y_train, y_test, "Multiple Linear Regression (Normal Equation)")

    # Print coefficients
    print("\nModel Coefficients:")
    coeffs = mlr_model.get_coefficients(feature_names)
    for name, value in coeffs.items():
        print(f"  {name}: {value:.4f}")

    # Train Multiple Linear Regression (Gradient Descent)
    print("\n" + "="*50)
    print("Training Multiple Linear Regression (Gradient Descent)...")
    print("="*50)

    mlr_gd_model = MultipleLinearRegressionGD(
        learning_rate=LEARNING_RATE,
        n_iterations=N_ITERATIONS
    )
    mlr_gd_model.fit(X_train_scaled, y_train)

    # Diagnostic output for GD convergence
    print(f"\n[Diagnostics]")
    print(f"  Initial loss: {mlr_gd_model.loss_history[0]:.6f}")
    print(f"  Final loss: {mlr_gd_model.loss_history[-1]:.6f}")
    print(f"  Iterations completed: {len(mlr_gd_model.loss_history)}/{N_ITERATIONS}")
    if len(mlr_gd_model.loss_history) > 10:
        recent_decrease = mlr_gd_model.loss_history[-10] - mlr_gd_model.loss_history[-1]
        print(f"  Loss decrease (last 10 iters): {recent_decrease:.6f}")
        if recent_decrease > 0.001:
            print(f"  ⚠ WARNING: Loss still decreasing significantly - may need more iterations")

    mlr_gd_results = evaluate_model(mlr_gd_model, X_train_scaled, X_test_scaled,
                                    y_train, y_test, "Multiple Linear Regression (Gradient Descent)")

    # Print coefficients
    print("\nModel Coefficients:")
    coeffs_gd = mlr_gd_model.get_coefficients(feature_names)
    for name, value in coeffs_gd.items():
        print(f"  {name}: {value:.4f}")

    # Save models
    print("\nSaving models...")
    import pickle
    with open(MODELS_DIR / 'multiple_regression.pkl', 'wb') as f:
        pickle.dump(mlr_model, f)
    print(f"Model saved to {MODELS_DIR / 'multiple_regression.pkl'}")
    
    with open(MODELS_DIR / 'multiple_regression_gd.pkl', 'wb') as f:
        pickle.dump(mlr_gd_model, f)
    print(f"Model saved to {MODELS_DIR / 'multiple_regression_gd.pkl'}")

    # Plot loss curve
    print("\nPlotting loss curve...")
    plot_path = PLOTS_DIR / 'multiple_regression_gd_loss.png'
    mlr_gd_model.plot_loss(save_path=str(plot_path))

    # Compare results
    print("\n" + "="*50)
    print("Model Comparison")
    print("="*50)
    print(f"{'Metric':<20} {'Normal Eq':<15} {'Gradient Descent':<15}")
    print("-"*50)
    print(f"{'Test MSE':<20} {mlr_results['test_mse']:<15.4f} {mlr_gd_results['test_mse']:<15.4f}")
    print(f"{'Test R²':<20} {mlr_results['test_r2']:<15.4f} {mlr_gd_results['test_r2']:<15.4f}")
    print(f"{'Test MAE':<20} {mlr_results['test_mae']:<15.4f} {mlr_gd_results['test_mae']:<15.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train Multiple Linear Regression models')
    parser.add_argument('--data', type=str, required=True, help='Path or filename of dataset (CSV or Parquet). If not absolute, looks inside the project data folder.')
    parser.add_argument('--target', type=str, default=None, help='Optional target column name (numeric). If omitted, uses demand_count if present or last column.')

    args = parser.parse_args()
    main(args.data, target_column=args.target)
