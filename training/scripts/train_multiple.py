import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

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


def _one_hot_objects(df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not obj_cols:
        return df
    return pd.get_dummies(df, columns=obj_cols, drop_first=True)


def load_data(file_path, target_column: str | None = None):
    """
    Load data from CSV or Parquet. Prefer a correct time-series target if present:

    - If 'demand_target' exists, use it as y (next-hour demand).
      Drop ['demand_target', 'demand_count', 'datetime_hour'] from X to avoid leakage.
    - Else if 'demand_count' exists, use it as y (current-hour demand).
      Drop ['demand_count', 'datetime_hour'] from X.
    - Else if target_column is provided, use that as y and drop it from X.
    - Else fall back to: use all numeric columns, last numeric as y.

    Returns (X, y, feature_names) as numpy arrays.
    """
    dataset_path = _resolve_dataset_path(file_path)
    if dataset_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(dataset_path)
    elif dataset_path.suffix.lower() == ".csv":
        # Parse datetime if present
        df = pd.read_csv(dataset_path, parse_dates=["datetime_hour"], infer_datetime_format=True)
    else:
        raise ValueError(f"Unsupported file extension '{dataset_path.suffix}'. Use CSV or Parquet files.")

    # Sort chronologically if datetime exists (keeps temporal order later)
    if "datetime_hour" in df.columns:
        df = df.sort_values("datetime_hour").reset_index(drop=True)

    # Prefer demand_target (next-hour) for forecasting tasks if present
    if "demand_target" in df.columns:
        y = df["demand_target"].to_numpy()
        drop_cols = ["demand_target"]
        if "demand_count" in df.columns:
            drop_cols.append("demand_count")
        if "datetime_hour" in df.columns:
            drop_cols.append("datetime_hour")
        feature_df = df.drop(columns=drop_cols, errors="ignore")
        feature_df = _one_hot_objects(feature_df)
        numeric_df = feature_df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] == 0:
            raise ValueError("No numeric features remain after dropping target and time columns.")
        X = numeric_df.to_numpy()
        feature_names = numeric_df.columns.tolist()
        print("Detected 'demand_target' (next-hour demand). Dropped ['demand_target','demand_count','datetime_hour'] from features.")
        print(f"Using {X.shape[1]} numeric features.")
        return X, y, feature_names

    # Special case: if demand_target absent but demand_count present → use current-hour regression
    if "demand_count" in df.columns:
        y = df["demand_count"].to_numpy()
        drop_cols = ["demand_count"]
        if "datetime_hour" in df.columns:
            drop_cols.append("datetime_hour")
        feature_df = df.drop(columns=drop_cols, errors="ignore")
        feature_df = _one_hot_objects(feature_df)
        numeric_df = feature_df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] == 0:
            raise ValueError("After dropping 'demand_count' and 'datetime_hour' no numeric features remain.")
        X = numeric_df.to_numpy()
        feature_names = numeric_df.columns.tolist()
        print("Detected 'demand_count' (current-hour demand). Dropped ['demand_count','datetime_hour'] from features.")
        print(f"Using {X.shape[1]} numeric features.")
        return X, y, feature_names

    # If explicit target specified
    if target_column:
        if target_column not in df.columns:
            raise KeyError(f"Target column '{target_column}' not found. Available columns: {df.columns.tolist()}")
        y = df[target_column].to_numpy()
        if not np.issubdtype(y.dtype, np.number):
            raise ValueError(f"Target column '{target_column}' must be numeric for regression. Got dtype {df[target_column].dtype}")
        feature_df = df.drop(columns=[target_column])
        feature_df = _one_hot_objects(feature_df)
        numeric_df = feature_df.select_dtypes(include=[np.number])
        X = numeric_df.to_numpy()
        feature_names = numeric_df.columns.tolist()
        print(f"Using target column '{target_column}' and {X.shape[1]} numeric features")
        return X, y, feature_names

    # Fallback: numeric-only, last numeric is target
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        raise ValueError(f"Need at least 2 numeric columns. Found {numeric_df.shape[1]}")
    X = numeric_df.iloc[:, :-1].to_numpy()
    y = numeric_df.iloc[:, -1].to_numpy()
    feature_names = numeric_df.columns[:-1].tolist()
    print(f"Selected {X.shape[1]} numeric features from {df.shape[1]} total columns")
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

    # Create forecast plot with rolling mean
    forecast_plot_path = PLOTS_DIR / f"{model_name.lower().replace(' ', '_')}_forecast_vs_actual_rolling_full.png"
    N = len(y_test)
    W = 10  # rolling window (hours)

    yt = pd.Series(y_test[:N]).reset_index(drop=True)
    yp = pd.Series(y_test_pred[:N]).reset_index(drop=True)

    plt.figure(figsize=(12, 5))
    plt.plot(yt, label='Actual', color='blue', alpha=0.35)
    plt.plot(yp, label='Predicted', color='orange', alpha=0.35, linestyle='--')

    plt.plot(yt.rolling(W, min_periods=1).mean(), label=f'Actual {W}h MA', color='blue')
    plt.plot(yp.rolling(W, min_periods=1).mean(), label=f'Pred {W}h MA', color='orange', linestyle='--')

    plt.title(f"Forecast vs Actual (First {N} Test Points) with {W}h Rolling Mean")
    plt.xlabel("Time (Test Index)")
    plt.ylabel("Demand Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(forecast_plot_path, dpi=300)
    plt.close()
    print(f"Saved forecast plot to {forecast_plot_path}")

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

    # Time-aware holdout: preserve order (no shuffling)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Data diagnostics
    print(f"\n{'='*60}")
    print("DATA DIAGNOSTICS")
    print(f"{'='*60}")
    print(f"Dataset shape: {X.shape} ({X.shape[0]} samples, {X.shape[1]} features)")
    print(f"Features: {feature_names[:5]}..." if len(feature_names) > 5 else f"Features: {feature_names}")
    print(f"Target variable:")
    print(f"  Range: [{y.min():.4f}, {y.max():.4f}]")
    print(f"  Mean: {y.mean():.4f}, Std: {y.std():.4f}")
    print(f"  Variance: {y.var():.4f}")

    # Check for NaN/Inf
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print(f"  ⚠ WARNING: X contains NaN or Inf values!")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        print(f"  ⚠ WARNING: y contains NaN or Inf values!")

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"{'='*60}")

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

    # Baseline comparison
    print(f"\n{'='*50}")
    print("BASELINE COMPARISON")
    print(f"{'='*50}")
    y_mean_pred = np.full_like(y_test, y_train.mean())
    baseline_mse = mean_squared_error(y_test, y_mean_pred)
    baseline_r2 = r2_score(y_test, y_mean_pred)
    print(f"Predicting mean ({y_train.mean():.4f}):")
    print(f"  Baseline MSE: {baseline_mse:.4f}")
    print(f"  Baseline R²: {baseline_r2:.4f}")
    print(f"Model improvement: {(1 - mlr_gd_results['test_mse']/baseline_mse)*100:.2f}% better than baseline")

    # Feature importance analysis
    print(f"\n{'='*50}")
    print("FEATURE ANALYSIS")
    print(f"{'='*50}")

    coeffs_gd = mlr_gd_model.get_coefficients(feature_names)
    sorted_coeffs = sorted(coeffs_gd.items(), key=lambda x: abs(x[1]), reverse=True)

    print(f"\nTop 10 features by |coefficient|:")
    for i, (feat, coef) in enumerate(sorted_coeffs[:10], 1):
        if feat != 'intercept':
            print(f"  {i}. {feat}: {coef:.6f}")

    # Count near-zero coefficients
    coeff_values = [v for k, v in coeffs_gd.items() if k != 'intercept']
    near_zero = sum(1 for coef in coeff_values if abs(coef) < 1e-3)
    print(f"\nFeatures with near-zero coefficients (<0.001): {near_zero}/{len(feature_names)}")

    if near_zero > len(feature_names) * 0.8:
        print(f"⚠ WARNING: {near_zero} features have near-zero coefficients!")
        print(f"  This suggests most features are not predictive for the target.")

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
    parser.add_argument('--data', type=str, required=True,
                        help='Path or filename of dataset (CSV or Parquet). If not absolute, looks inside the project data folder.')
    parser.add_argument('--target', type=str, default=None,
                        help='Optional target column name (numeric). If omitted, will use demand_target if present, else demand_count, else last numeric column.')

    args = parser.parse_args()
    main(args.data, target_column=args.target)
