"""Training script for from-scratch clustering algorithms (K-Means & Agglomerative)."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

SCRIPT_PATH = Path(__file__).resolve()
SYS_PATH_ROOT = SCRIPT_PATH.parents[2]
if str(SYS_PATH_ROOT) not in sys.path:
    sys.path.append(str(SYS_PATH_ROOT))

DATA_DIR = SYS_PATH_ROOT / "data"
OUTPUT_DIR = SYS_PATH_ROOT / "output"
REPORTS_DIR = OUTPUT_DIR

from training.classification.clustering import (  # noqa: E402
    AgglomerativeClusteringScratch,
    KMeansScratch,
)


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


def load_dataset(
    data_path: str,
    target_column: Optional[str],
) -> tuple[np.ndarray, Optional[np.ndarray], list[str], pd.DataFrame]:
    dataset_path = _resolve_dataset_path(data_path)
    if dataset_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(dataset_path)
    elif dataset_path.suffix.lower() == ".csv":
        df = pd.read_csv(dataset_path)
    else:
        raise ValueError(
            f"Unsupported file extension '{dataset_path.suffix}'. Use CSV or Parquet files."
        )

    y: Optional[np.ndarray] = None
    if target_column:
        if target_column not in df.columns:
            raise KeyError(
                f"Target column '{target_column}' not found. Available columns: {df.columns.tolist()}"
            )
        target_series = (
            df[target_column]
            .astype("string")
            .fillna("")
            .str.strip()
            .str.lower()
        )
        y_mapped = np.where(target_series == "completed", "Completed", "Other")
        y = y_mapped.astype(str)
        feature_df = df.drop(columns=[target_column])
    else:
        feature_df = df

    X = feature_df.to_numpy(dtype=np.float32)
    feature_names = feature_df.columns.tolist()
    return X, y, feature_names, feature_df.copy()


def _cluster_size_summary(labels: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(labels, return_counts=True)
    return {str(label): int(count) for label, count in zip(unique, counts)}


def _safe_metric(
    metric_fn,
    X: np.ndarray,
    labels: np.ndarray,
    *,
    random_state: Optional[int] = None,
    max_samples: int = 2000,
) -> Optional[float]:
    try:
        unique_labels = np.unique(labels)
        if unique_labels.size < 2 or unique_labels.size >= len(labels):
            return None
        X_evaluate = X
        labels_evaluate = labels
        if max_samples > 0 and X.shape[0] > max_samples:
            rng = np.random.default_rng(random_state)
            sample_indices = rng.choice(X.shape[0], max_samples, replace=False)
            X_evaluate = X[sample_indices]
            labels_evaluate = labels[sample_indices]
        return float(metric_fn(X_evaluate, labels_evaluate))
    except Exception:  # pragma: no cover - defensive safeguard
        return None


def _evaluate_against_labels(
    y_true: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    y_cast = np.asarray(y_true)
    if y_cast.ndim != 1:
        y_cast = y_cast.reshape(-1)
    y_cast = y_cast.astype(str)
    return {
        "adjusted_rand_index": float(adjusted_rand_score(y_cast, labels)),
        "normalized_mutual_info": float(normalized_mutual_info_score(y_cast, labels)),
    }


def _majority_label_mapping(cluster_labels: np.ndarray, y_true: np.ndarray) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for cluster in np.unique(cluster_labels):
        mask = cluster_labels == cluster
        values, counts = np.unique(y_true[mask], return_counts=True)
        if values.size == 0:
            continue
        majority_label = values[np.argmax(counts)]
        mapping[int(cluster)] = str(majority_label)
    return mapping


def _classify_clusters(
    cluster_labels: np.ndarray,
    y_true: np.ndarray,
    *,
    class_order: Optional[list[str]] = None,
) -> tuple[dict[int, str], np.ndarray, dict[str, Any]]:
    mapping = _majority_label_mapping(cluster_labels, y_true)
    predicted_labels = np.vectorize(lambda label: mapping.get(int(label), "Unknown"))(cluster_labels)

    if class_order is None:
        class_order = [str(cls) for cls in np.unique(y_true)]

    report_dict = classification_report(
        y_true,
        predicted_labels,
        labels=class_order,
        target_names=class_order,
        zero_division=0,
        output_dict=True,
    )
    conf = confusion_matrix(y_true, predicted_labels, labels=class_order)

    metrics = {
        "accuracy": float(accuracy_score(y_true, predicted_labels)),
        "precision_macro": float(precision_score(y_true, predicted_labels, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, predicted_labels, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, predicted_labels, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, predicted_labels, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, predicted_labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, predicted_labels, average="weighted", zero_division=0)),
        "confusion_matrix": conf.tolist(),
        "classification_report": report_dict,
        "cluster_label_mapping": {str(cluster): label for cluster, label in mapping.items()},
    }
    return mapping, predicted_labels, metrics


def run_kmeans(
    X: np.ndarray,
    n_clusters: int,
    random_state: Optional[int],
) -> tuple[KMeansScratch, dict[str, Any], np.ndarray]:
    model = KMeansScratch(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(X)
    metrics = {
        "n_clusters": n_clusters,
        "n_iter": model.n_iter_,
        "inertia": float(model.inertia_ if model.inertia_ is not None else np.nan),
        "cluster_sizes": _cluster_size_summary(labels),
    }
    if model.cluster_centers_ is not None:
        metrics["cluster_centers"] = model.cluster_centers_.tolist()
    return model, metrics, labels


def run_agglomerative(
    X: np.ndarray,
    n_clusters: int,
    linkage: str,
) -> tuple[AgglomerativeClusteringScratch, dict[str, Any], np.ndarray]:
    model = AgglomerativeClusteringScratch(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)
    metrics = {
        "n_clusters": int(np.unique(labels).size),
        "cluster_sizes": _cluster_size_summary(labels),
        "linkage": linkage,
        "merge_history_length": len(model.merge_history_),
    }
    if model.distance_history_:
        metrics["last_merge_distance"] = float(model.distance_history_[-1])
    return model, metrics, labels


def main(args: argparse.Namespace) -> None:
    print("Loading dataset...")
    X_raw, y_raw, feature_names, feature_df = load_dataset(args.data, args.target)

    scaler = StandardScaler() if args.scale else None
    X = scaler.fit_transform(X_raw) if scaler is not None else X_raw
    rng = np.random.default_rng(args.random_state)

    label_array: Optional[np.ndarray] = None
    label_classes: Optional[list[str]] = None
    if y_raw is not None:
        label_array = np.asarray(y_raw, dtype=str)
        label_classes = ["Other", "Completed"]

    results: dict[str, Any] = {
        "dataset": args.data,
        "feature_names": feature_names,
        "scaled": bool(args.scale),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "agglomerative_max_samples": args.agglomerative_max_samples,
    }
    if label_classes is not None:
        results["label_classes"] = label_classes

    assignments_df = feature_df.copy()
    if label_array is not None:
        assignments_df["true_label"] = label_array

    if args.algorithm in {"kmeans", "both"}:
        print("Running K-Means clustering...")
        kmeans_model, kmeans_metrics, kmeans_labels = run_kmeans(
            X, args.n_clusters, args.random_state
        )
        assignments_df["kmeans_label"] = kmeans_labels
        kmeans_metrics["silhouette"] = _safe_metric(
            silhouette_score, X, kmeans_labels, random_state=args.random_state
        )
        kmeans_metrics["davies_bouldin"] = _safe_metric(
            davies_bouldin_score, X, kmeans_labels, random_state=args.random_state
        )
        kmeans_metrics["calinski_harabasz"] = _safe_metric(
            calinski_harabasz_score, X, kmeans_labels, random_state=args.random_state
        )
        if label_array is not None:
            kmeans_metrics.update(_evaluate_against_labels(label_array, kmeans_labels))
            mapping, predicted_labels, clf_metrics = _classify_clusters(
                kmeans_labels,
                label_array,
                class_order=label_classes,
            )
            assignments_df["kmeans_predicted_label"] = predicted_labels
            kmeans_metrics["classification_evaluation"] = clf_metrics

            print("\nK-Means classification-style evaluation (majority vote):")
            print("=" * 50)
            print(f"Accuracy: {clf_metrics['accuracy']:.4f}")
            print(f"Macro F1: {clf_metrics['f1_macro']:.4f}")
            print(f"Weighted F1: {clf_metrics['f1_weighted']:.4f}")
            print("Cluster → Label mapping:")
            for cluster, label in mapping.items():
                print(f"  Cluster {cluster}: {label}")
            conf_matrix = np.array(clf_metrics["confusion_matrix"], dtype=int)
            print("\nConfusion Matrix:")
            print(conf_matrix)
            print("\nClassification Report:")
            print(
                classification_report(
                    label_array,
                    predicted_labels,
                    labels=label_classes,
                    target_names=label_classes,
                    zero_division=0,
                )
            )
        results["kmeans"] = kmeans_metrics

    if args.algorithm in {"agglomerative", "both"}:
        print(f"Running Agglomerative clustering (linkage='{args.linkage}')...")

        limit = args.agglomerative_max_samples or 0
        sample_size = X.shape[0]
        if limit > 0 and X.shape[0] > limit:
            sample_size = limit

        sample_indices = np.arange(X.shape[0])
        if sample_size < X.shape[0]:
            sample_indices = np.sort(
                rng.choice(X.shape[0], sample_size, replace=False)
            )

        X_ag = X[sample_indices]
        ag_model, ag_metrics, ag_labels_subset = run_agglomerative(
            X_ag, args.n_clusters, args.linkage
        )

        full_labels = np.empty(X.shape[0], dtype=int)
        full_labels[sample_indices] = ag_labels_subset

        if sample_indices.size < X.shape[0]:
            unique_clusters = np.unique(ag_labels_subset)
            centroids = np.vstack(
                [X_ag[ag_labels_subset == cluster].mean(axis=0) for cluster in unique_clusters]
            )
            remaining_indices = np.setdiff1d(np.arange(X.shape[0]), sample_indices, assume_unique=True)
            distances = np.linalg.norm(
                X[remaining_indices][:, None, :] - centroids[None, :, :], axis=2
            )
            nearest_clusters = unique_clusters[np.argmin(distances, axis=1)]
            full_labels[remaining_indices] = nearest_clusters
            ag_metrics["sample_size"] = int(sample_indices.size)
            ag_metrics["assigned_via_centroid"] = int(remaining_indices.size)
        else:
            ag_metrics["sample_size"] = int(sample_indices.size)

        ag_labels = full_labels
        ag_metrics["n_clusters"] = int(np.unique(ag_labels).size)
        assignments_df["agglomerative_label"] = ag_labels
        ag_metrics["silhouette"] = _safe_metric(
            silhouette_score, X, ag_labels, random_state=args.random_state
        )
        ag_metrics["davies_bouldin"] = _safe_metric(
            davies_bouldin_score, X, ag_labels, random_state=args.random_state
        )
        ag_metrics["calinski_harabasz"] = _safe_metric(
            calinski_harabasz_score, X, ag_labels, random_state=args.random_state
        )
        if label_array is not None:
            ag_metrics.update(_evaluate_against_labels(label_array, ag_labels))
            mapping, predicted_labels, clf_metrics = _classify_clusters(
                ag_labels,
                label_array,
                class_order=label_classes,
            )
            assignments_df["agglomerative_predicted_label"] = predicted_labels
            ag_metrics["classification_evaluation"] = clf_metrics

            print("\nAgglomerative classification-style evaluation (majority vote):")
            print("=" * 50)
            print(f"Accuracy: {clf_metrics['accuracy']:.4f}")
            print(f"Macro F1: {clf_metrics['f1_macro']:.4f}")
            print(f"Weighted F1: {clf_metrics['f1_weighted']:.4f}")
            print("Cluster → Label mapping:")
            for cluster, label in mapping.items():
                print(f"  Cluster {cluster}: {label}")
            conf_matrix = np.array(clf_metrics["confusion_matrix"], dtype=int)
            print("\nConfusion Matrix:")
            print(conf_matrix)
            print("\nClassification Report:")
            print(
                classification_report(
                    label_array,
                    predicted_labels,
                    labels=label_classes,
                    target_names=label_classes,
                    zero_division=0,
                )
            )
        results["agglomerative"] = ag_metrics

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = Path(args.report_path) if args.report_path else OUTPUT_DIR / "clustering_metrics.json"
    assignments_path = OUTPUT_DIR / "clustering_assignments.parquet"

    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    print(f"Metrics summary saved to {report_path}")

    assignments_df.to_parquet(assignments_path, index=False)
    print(f"Cluster assignments saved to {assignments_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run from-scratch clustering algorithms on tabular datasets.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path or filename of dataset (CSV or Parquet). If not absolute, looks inside the project data folder.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Optional target column used only for external cluster evaluation metrics.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=3,
        help="Number of clusters to form (shared by both algorithms).",
    )
    parser.add_argument(
        "--linkage",
        choices=["average", "single", "complete", "ward"],
        default="average",
        help="Linkage criterion for agglomerative clustering.",
    )
    parser.add_argument(
        "--algorithm",
        choices=["kmeans", "agglomerative", "both"],
        default="both",
        help="Which algorithms to execute.",
    )
    parser.add_argument(
        "--agglomerative-max-samples",
        type=int,
        default=1500,
        help=(
            "Maximum number of samples used when fitting the agglomerative model. "
            "Set to 0 to disable downsampling."
        ),
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        help="Apply standard scaling to features before clustering.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for centroid initialisation in K-Means.",
    )
    parser.add_argument(
        "--report-path",
        help="Optional custom path for the JSON metrics report.",
    )
    main(parser.parse_args())
