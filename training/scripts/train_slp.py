"""Training script for Single Layer Perceptron built from scratch."""

import argparse
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
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

# Try to import SMOTE, with fallback if not available
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: imbalanced-learn not available. Install with: pip install imbalanced-learn")

# Ensure project root is on the path when executing as a script
SCRIPT_PATH = Path(__file__).resolve()
SYS_PATH_ROOT = SCRIPT_PATH.parents[2]
if str(SYS_PATH_ROOT) not in sys.path:
    sys.path.append(str(SYS_PATH_ROOT))

DATA_DIR = SYS_PATH_ROOT / "data"
OUTPUT_DIR = SYS_PATH_ROOT / "output"
MODELS_DIR = OUTPUT_DIR / "models"
PLOTS_DIR = OUTPUT_DIR / "plots"

from training.classification.single_layer_perceptron import SLPFromScratch  # noqa: E402

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Parameter grid for SLP hyperparameter search
SLP_PARAM_GRID = {
    "learning_rate": [0.001, 0.01, 0.1],
    "num_iterations": [500, 1000, 2000],
}


def apply_resampling(X: np.ndarray, y: np.ndarray, method: str = "none", random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Apply resampling techniques to handle class imbalance.
    
    Args:
        X: Feature matrix
        y: Target labels
        method: Resampling method ('none', 'smote', 'random_oversample')
        random_state: Random state for reproducibility
        
    Returns:
        Resampled X and y arrays
    """
    if method == "none":
        return X, y
    
    if not SMOTE_AVAILABLE:
        print(f"Warning: {method} resampling requested but imbalanced-learn not available. Skipping resampling.")
        return X, y
    
    print(f"Applying {method} resampling...")
    
    if method == "smote":
        resampler = SMOTE(random_state=random_state)
    elif method == "random_oversample":
        resampler = RandomOverSampler(random_state=random_state)
    else:
        print(f"Unknown resampling method: {method}. Available methods: none, smote, random_oversample")
        return X, y
    
    X_resampled, y_resampled = resampler.fit_resample(X, y)
    return X_resampled, y_resampled


def load_data(file_path: str, target_column: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load and preprocess data from parquet file."""
    df = pd.read_parquet(DATA_DIR / file_path)
    
    # Separate features and target
    X = df.drop(columns=[target_column]).values
    y_labels = df[target_column].values
    
    # Convert to binary classification: Completed vs Other
    # Map "Completed" to 1, everything else to 0
    y_binary = np.where(y_labels == "Completed", 1, 0)
    classes = ["Other", "Completed"]  # 0: Other, 1: Completed
    
    return X, y_binary, classes


def perform_slp_grid_search(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
    scoring: str = "accuracy",
    pca_components: int = None,
    use_cuda: bool = False,
) -> tuple[dict, list, float]:
    """Perform grid search for SLP hyperparameters."""
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)
    results = []
    best_score = -np.inf
    best_params = None
    
    # Generate all parameter combinations
    param_combinations = []
    for lr in SLP_PARAM_GRID["learning_rate"]:
        for n_iter in SLP_PARAM_GRID["num_iterations"]:
            param_combinations.append({
                "learning_rate": lr,
                "num_iterations": n_iter,
            })
    
    print(f"Testing {len(param_combinations)} parameter combinations for SLP...")
    
    for i, params in enumerate(param_combinations):
        print(f"Progress: {i+1}/{len(param_combinations)} parameter combinations")
        
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            try:
                X_train_fold = X[train_idx]
                X_val_fold = X[val_idx]
                y_train_fold = y[train_idx]
                y_val_fold = y[val_idx]
                
                # Standardize features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_fold)
                X_val_scaled = scaler.transform(X_val_fold)
                
                # Apply PCA if specified
                if pca_components is not None:
                    pca = PCA(n_components=pca_components, random_state=random_state)
                    X_train_scaled = pca.fit_transform(X_train_scaled)
                    X_val_scaled = pca.transform(X_val_scaled)
                
                # Create and train SLP
                slp = SLPFromScratch(
                    input_size=X_train_scaled.shape[1],
                    learning_rate=params["learning_rate"],
                    num_iterations=params["num_iterations"],
                    use_cuda=use_cuda
                )
                
                slp.fit(X_train_scaled, y_train_fold)
                
                # Make predictions
                y_pred = slp.predict(X_val_scaled)
                
                # Calculate score
                if scoring == "accuracy":
                    score = accuracy_score(y_val_fold, y_pred)
                elif scoring == "f1":
                    score = f1_score(y_val_fold, y_pred, average="binary")
                elif scoring == "macro_f1":
                    score = f1_score(y_val_fold, y_pred, average="macro")
                else:
                    score = accuracy_score(y_val_fold, y_pred)
                
                scores.append(score)
                
            except Exception as e:
                print(f"Error in fold {fold+1} with params {params}: {e}")
                continue
        
        if scores:  # Only process if we have valid scores
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            results.append({
                "params": params.copy(),
                "mean_score": mean_score,
                "std_score": std_score,
                "individual_scores": scores.copy()
            })
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params.copy()
        else:
            print(f"All folds failed for params {params}, skipping...")
    
    if not results:
        raise RuntimeError("Grid search failed to evaluate any parameter combinations.")
    
    return best_params, results, best_score


def predict_with_threshold(model, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Make predictions with custom threshold."""
    probabilities = model.predict_proba(X)
    return (probabilities > threshold).astype(int)


def evaluate(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
    model_name: str,
    threshold: float = 0.5,
) -> dict:
    """Evaluate model performance on train and test sets."""
    
    # Make predictions with threshold
    y_train_pred = predict_with_threshold(model, X_train, threshold)
    y_test_pred = predict_with_threshold(model, X_test, threshold)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    train_f1_macro = f1_score(y_train, y_train_pred, average="macro", zero_division=0)
    test_f1_macro = f1_score(y_test, y_test_pred, average="macro", zero_division=0)
    
    # Per-class metrics
    train_precision = precision_score(y_train, y_train_pred, average=None, zero_division=0)
    train_recall = recall_score(y_train, y_train_pred, average=None, zero_division=0)
    test_precision = precision_score(y_test, y_test_pred, average=None, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average=None, zero_division=0)
    
    # Confusion matrices
    train_cm = confusion_matrix(y_train, y_train_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)
    
    metrics = {
        "model_name": model_name,
        "threshold": threshold,
        "train": {
            "accuracy": train_accuracy,
            "f1_macro": train_f1_macro,
            "precision_per_class": train_precision.tolist(),
            "recall_per_class": train_recall.tolist(),
            "confusion_matrix": train_cm.tolist(),
        },
        "test": {
            "accuracy": test_accuracy,
            "f1_macro": test_f1_macro,
            "precision_per_class": test_precision.tolist(),
            "recall_per_class": test_recall.tolist(),
            "confusion_matrix": test_cm.tolist(),
        },
        "classes": classes,
    }
    
    # Print results
    print(f"\n{model_name} Results (threshold={threshold:.3f}):")
    print("=" * 60)
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy:  {test_accuracy:.4f}")
    print(f"Train F1 (macro): {train_f1_macro:.4f}")
    print(f"Test F1 (macro):  {test_f1_macro:.4f}")
    
    print("\nPer-class Performance:")
    for i, class_name in enumerate(classes):
        print(f"  {class_name}:")
        print(f"    Train - Precision: {train_precision[i]:.4f}, Recall: {train_recall[i]:.4f}")
        print(f"    Test  - Precision: {test_precision[i]:.4f}, Recall: {test_recall[i]:.4f}")
    
    print(f"\nTest Confusion Matrix:")
    print(test_cm)
    
    # Generate classification report
    label_indices = list(range(len(classes)))
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


def plot_training_loss(losses: list, save_path: Path) -> None:
    """Plot and save training loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Single Layer Perceptron Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to: {save_path}")


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
    train_distribution_named = {
        classes[int(label)]: int(count) for label, count in zip(unique, counts)
    }
    print(f"Class distribution (train): {train_distribution_named}")
    
    # Apply resampling if requested
    X_train_resampled, y_train_resampled = apply_resampling(
        X_train, y_train, method=args.resampling, random_state=RANDOM_STATE
    )
    
    # Update distribution info after resampling
    if args.resampling != "none":
        unique_resampled, counts_resampled = np.unique(y_train_resampled.astype(int), return_counts=True)
        train_distribution_resampled = {
            classes[int(label)]: int(count) for label, count in zip(unique_resampled, counts_resampled)
        }
        print(f"Class distribution (after {args.resampling}): {train_distribution_resampled}")
        X_train_for_cv = X_train_resampled
        y_train_for_cv = y_train_resampled
    else:
        X_train_for_cv = X_train
        y_train_for_cv = y_train
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA if specified
    if args.pca_components is not None:
        print(f"Applying PCA with {args.pca_components} components...")
        pca = PCA(n_components=args.pca_components, random_state=RANDOM_STATE)
        X_train_proc = pca.fit_transform(X_train_scaled)
        X_test_proc = pca.transform(X_test_scaled)
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    else:
        X_train_proc = X_train_scaled
        X_test_proc = X_test_scaled
    
    # Hyperparameter search
    print("\n" + "=" * 50)
    print("Hyperparameter search (Single Layer Perceptron)...")
    print("=" * 50)
    
    best_params, cv_results, best_cv_score = perform_slp_grid_search(
        X_train_for_cv, y_train_for_cv, RANDOM_STATE, scoring=args.cv_metric, pca_components=args.pca_components, use_cuda=args.use_cuda
    )
    best_cv_entry = max(cv_results, key=lambda entry: entry["mean_score"])
    
    print(
        "Best parameters: "
        f"learning_rate={best_params['learning_rate']}, "
        f"num_iterations={best_params['num_iterations']}"
    )
    
    print(
        f"Cross-validated {args.cv_metric}: "
        f"{best_cv_entry['mean_score']:.4f} ± {best_cv_entry['std_score']:.4f}"
    )
    
    # Train final model
    print("\n" + "=" * 50)
    print("Training Single Layer Perceptron (Completed vs Other)...")
    print("=" * 50)
    
    final_params = {
        "input_size": X_train_proc.shape[1],
        "learning_rate": float(best_params["learning_rate"]),
        "num_iterations": int(best_params["num_iterations"]),
        "use_cuda": args.use_cuda,
    }
    
    slp_model = SLPFromScratch(**final_params)
    weights, bias, losses = slp_model.fit(X_train_proc, y_train_resampled)
    
    # Determine optimal threshold if not provided
    if args.threshold is None:
        # Find threshold that balances precision and recall for minority class
        probabilities = slp_model.predict_proba(X_train_proc)
        thresholds = np.percentile(probabilities, [10, 20, 30, 40, 50, 60, 70, 80, 90])
        best_threshold = 0.5
        best_f1 = 0.0
        
        print("\nSearching for optimal threshold...")
        for thresh in thresholds:
            y_pred_thresh = (probabilities > thresh).astype(int)
            f1_macro = f1_score(y_train_resampled, y_pred_thresh, average="macro", zero_division=0)
            print(f"Threshold {thresh:.3f}: macro F1 = {f1_macro:.4f}")
            if f1_macro > best_f1:
                best_f1 = f1_macro
                best_threshold = thresh
        
        final_threshold = best_threshold
        print(f"Selected threshold: {final_threshold:.3f} (macro F1: {best_f1:.4f})")
    else:
        final_threshold = args.threshold
        print(f"Using provided threshold: {final_threshold}")
    
    model_name = "Single Layer Perceptron (Completed vs Other)"
    metrics = evaluate(
        slp_model,
        X_train_proc,
        y_train_resampled,
        X_test_proc,
        y_test,
        classes,
        model_name,
        threshold=final_threshold,
    )
    
    # Plot training loss
    loss_plot_path = PLOTS_DIR / "slp_training_loss.png"
    plot_training_loss(losses, loss_plot_path)
    
    # Save metrics
    metrics_path = OUTPUT_DIR / "slp_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    # Save model parameters
    model_save_path = MODELS_DIR / "slp_model.npz"
    np.savez(
        model_save_path,
        weights=weights,
        bias=bias,
        threshold=final_threshold,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        pca_components=pca.components_ if args.pca_components is not None else None,
        pca_mean=pca.mean_ if args.pca_components is not None else None,
        classes=classes,
        params=final_params,
    )
    print(f"Model saved to: {model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Single Layer Perceptron from scratch")
    parser.add_argument("--data", type=str, required=True, help="Input parquet file name")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    parser.add_argument("--pca-components", type=int, help="Number of PCA components (optional)")
    parser.add_argument(
        "--cv-metric",
        type=str,
        default="f1",
        choices=["accuracy", "f1", "macro_f1"],
        help="Cross-validation scoring metric (f1 recommended for binary classification)",
    )
    parser.add_argument(
        "--resampling",
        type=str,
        default="none",
        choices=["none", "smote", "random_oversample"],
        help="Resampling method for class imbalance",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Custom decision threshold (if not provided, will be optimized)",
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="Use CUDA GPU acceleration if available",
    )
    
    main(parser.parse_args())
