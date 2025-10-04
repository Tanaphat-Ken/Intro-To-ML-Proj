"""Training script for Support Vector Machine classifiers built from scratch."""

import argparse
import json
import os
import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

# Ensure project root is on the path when executing as a script
SCRIPT_PATH = Path(__file__).resolve()
SYS_PATH_ROOT = SCRIPT_PATH.parents[2]
if str(SYS_PATH_ROOT) not in sys.path:
    sys.path.append(str(SYS_PATH_ROOT))

DATA_DIR = SYS_PATH_ROOT / "data"
OUTPUT_DIR = SYS_PATH_ROOT / "output"
MODELS_DIR = OUTPUT_DIR / "models"
PLOTS_DIR = OUTPUT_DIR / "plots"

from training.classification.support_vector_machine import (  # noqa: E402
    LinearSVMClassifier,
)


RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
LINEAR_CONFIG = dict(learning_rate=0.001, n_iterations=1500, C=1.0, batch_size=64)

SVM_PARAM_GRID = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "learning_rate": [0.01, 0.001],
    "n_iterations": [1000, 2000],
    "batch_size": [32, 64],
}
def _resolve_dataset_path(path_str: str) -> Path:
    """Resolve dataset path supporting files located inside the project `data` folder."""

    candidate = Path(path_str)

    potential_paths = [candidate]
    if not candidate.is_absolute():
        potential_paths.append(Path.cwd() / candidate)
        potential_paths.append(DATA_DIR / candidate)

    # If no suffix provided, try common ones
    if candidate.suffix == "":
        potential_paths.extend([p.with_suffix(ext) for p in potential_paths for ext in (".parquet", ".csv")])

    for path in potential_paths:
        if path.exists():
            return path

    tried = "\n".join(str(p) for p in potential_paths)
    raise FileNotFoundError(
        f"Could not locate dataset '{path_str}'. Paths tried:\n{tried}"
    )


def load_data(
    data_path: str,
    target_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load dataset, select target column, and return features with integer-encoded labels."""

    dataset_path = _resolve_dataset_path(data_path)

    if dataset_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(dataset_path)
    elif dataset_path.suffix.lower() == ".csv":
        df = pd.read_csv(dataset_path)
    else:
        raise ValueError(
            f"Unsupported file extension '{dataset_path.suffix}'. Use CSV or Parquet files."
        )

    if target_column is None:
        target_column = df.columns[-1]
    elif target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found. Available columns: {df.columns.tolist()}")

    feature_df = df.drop(columns=[target_column])

    target_series = (
        df[target_column]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
    )

    is_completed = target_series == "completed"
    mapped_labels = np.where(is_completed, "Completed", "Other")
    categorical_labels = pd.Categorical(
        mapped_labels, categories=["Other", "Completed"], ordered=True
    )

    classes = [str(c) for c in categorical_labels.categories]
    y = categorical_labels.codes.astype(np.int32)

    X = feature_df.to_numpy(dtype=np.float32)
    return X, y, classes


def _parameter_combinations(param_grid: dict[str, list]) -> list[dict[str, float | int]]:
    keys = list(param_grid.keys())
    values_product = product(*(param_grid[key] for key in keys))
    return [dict(zip(keys, combo)) for combo in values_product]


def perform_svm_grid_search(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
    scoring: str = "accuracy",
) -> tuple[dict[str, float | int], list[dict[str, float | int]], float]:
    """Evaluate SVM hyperparameters via stratified cross-validation."""

    if scoring not in {"accuracy", "macro_f1"}:
        raise ValueError("scoring must be either 'accuracy' or 'macro_f1'")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)
    results: list[dict[str, float | int]] = []
    best_params: dict[str, float | int] | None = None
    best_score = -np.inf

    for params in _parameter_combinations(SVM_PARAM_GRID):
        fold_scores: list[float] = []

        for train_idx, val_idx in skf.split(X, y):
            scaler = StandardScaler()
            X_train_fold = scaler.fit_transform(X[train_idx])
            X_val_fold = scaler.transform(X[val_idx])
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]

            svm = LinearSVMClassifier(
                learning_rate=float(params["learning_rate"]),
                n_iterations=int(params["n_iterations"]),
                C=float(params["C"]),
                batch_size=int(params["batch_size"]),
                random_state=random_state,
            )
            svm.fit(X_train_fold, y_train_fold)
            y_val_pred = svm.predict(X_val_fold)
            if scoring == "macro_f1":
                score = f1_score(y_val_fold, y_val_pred, average="macro", zero_division=0)
            else:
                score = accuracy_score(y_val_fold, y_val_pred)
            fold_scores.append(score)

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))

        result_entry: dict[str, float | int] = {
            **params,
            "mean_score": mean_score,
            "std_score": std_score,
            "metric": scoring,
        }
        results.append(result_entry)

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    if best_params is None:
        raise RuntimeError("Grid search failed to evaluate any parameter combinations.")

    return best_params, results, float(best_score)


def evaluate(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    classes: list[str],
    model_name: str,
) -> dict[str, float | dict]:
    """Compute multi-class classification metrics and print them to stdout."""

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "test_precision_macro": precision_score(
            y_test, y_test_pred, average="macro", zero_division=0
        ),
        "test_precision_weighted": precision_score(
            y_test, y_test_pred, average="weighted", zero_division=0
        ),
        "test_recall_macro": recall_score(
            y_test, y_test_pred, average="macro", zero_division=0
        ),
        "test_recall_weighted": recall_score(
            y_test, y_test_pred, average="weighted", zero_division=0
        ),
        "test_f1_macro": f1_score(y_test, y_test_pred, average="macro", zero_division=0),
        "test_f1_weighted": f1_score(
            y_test, y_test_pred, average="weighted", zero_division=0
        ),
    }

    print(f"\n{model_name} Results")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"{key.replace('_', ' ').title():<20}: {value:.4f}")

    print("\nConfusion Matrix:")
    label_indices = np.arange(len(classes))
    conf = confusion_matrix(y_test, y_test_pred, labels=label_indices)
    print(conf)

    report_str = classification_report(
        y_test,
        y_test_pred,
        labels=label_indices,
        target_names=classes,
        zero_division=0,
    )
    print("\nClassification Report:")
    print(report_str)

    metrics["classification_report"] = classification_report(
        y_test,
        y_test_pred,
        labels=label_indices,
        target_names=classes,
        zero_division=0,
        output_dict=True,
    )

    return metrics


def main(args: argparse.Namespace) -> None:
    print("Loading dataset...")
    X, y, classes = load_data(args.data, target_column=args.target)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Resolved classes: {classes}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Dataset shape: {X.shape}")
    print(f"Train/Test split: {X_train.shape[0]}/{X_test.shape[0]} samples")
    unique, counts = np.unique(y_train.astype(int), return_counts=True)
    train_distribution_indices = {
        int(label): int(count) for label, count in zip(unique, counts)
    }
    train_distribution_named = {
        classes[int(label)]: int(count) for label, count in zip(unique, counts)
    }
    print(f"Class distribution (train): {train_distribution_named}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Hyperparameter search (Linear SVM)...")
    print("=" * 50)

    best_params, cv_results, best_cv_score = perform_svm_grid_search(
        X_train, y_train, RANDOM_STATE, scoring=args.cv_metric
    )
    best_cv_entry = max(cv_results, key=lambda entry: entry["mean_score"])
    print(
        "Best parameters: "
        f"C={best_params['C']}, learning_rate={best_params['learning_rate']}, "
        f"n_iterations={best_params['n_iterations']}, batch_size={best_params['batch_size']}"
    )
    print(
        f"Cross-validated {args.cv_metric}: "
        f"{best_cv_entry['mean_score']:.4f} ± {best_cv_entry['std_score']:.4f}"
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Training Linear SVM (Completed vs Other)...")
    print("=" * 50)

    final_params = {
        "learning_rate": float(best_params["learning_rate"]),
        "n_iterations": int(best_params["n_iterations"]),
        "C": float(best_params["C"]),
        "batch_size": int(best_params["batch_size"]),
        "random_state": RANDOM_STATE,
    }

    linear_svm = LinearSVMClassifier(**final_params)
    linear_svm.fit(X_train_scaled, y_train)
    metrics = evaluate(
        linear_svm,
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        classes,
        "Linear SVM (Completed vs Other)",
    )

    loss_plot_path = PLOTS_DIR / "svm_linear_loss.png"
    linear_svm.plot_loss(save_path=str(loss_plot_path), show=args.plot)
    if args.plot:
        print(f"Displayed loss curves and saved to {loss_plot_path}")
    else:
        print(f"Loss curve saved to {loss_plot_path}")

    model_path = MODELS_DIR / "svm_linear_model.npz"
    np.savez(
        model_path,
        classes=np.array(classes),
        coefficients=linear_svm.coef_,
        intercept=np.array([linear_svm.intercept_], dtype=np.float32),
        loss_history=np.asarray(linear_svm.loss_history, dtype=np.float32),
    )
    print(f"Linear SVM parameters saved to {model_path}")

    # ------------------------------------------------------------------
    metrics_path = OUTPUT_DIR / "svm_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "classes": classes,
                "train_distribution_indices": train_distribution_indices,
                "train_distribution_labels": train_distribution_named,
                "linear_binary": metrics,
                "best_params": best_params,
                "cv_results": cv_results,
                "cv_folds": CV_FOLDS,
                "cv_metric": args.cv_metric,
                "best_cv_score": best_cv_score,
                "final_model_params": dict(
                    learning_rate=linear_svm.learning_rate,
                    n_iterations=linear_svm.n_iterations,
                    C=linear_svm.C,
                    fit_intercept=linear_svm.fit_intercept,
                    batch_size=linear_svm.batch_size,
                    random_state=linear_svm.random_state,
                ),
            },
            fp,
            indent=2,
        )
    print(f"Metrics summary saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train from-scratch SVM classifiers on a CSV or Parquet dataset")
    parser.add_argument(
        "--data",
        required=True,
        help="Path or filename of dataset (CSV or Parquet). If not absolute, looks inside the project data folder.",
    )
    parser.add_argument(
        "--target",
        help="Name of the target column. Defaults to the dataset's last column.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display optimisation curves for the trained models",
    )
    parser.add_argument(
        "--cv-metric",
        choices=["accuracy", "macro_f1"],
        default="accuracy",
        help="Metric optimised during cross-validation grid search",
    )
    main(parser.parse_args())
