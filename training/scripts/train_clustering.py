"""Training script for clustering-based classification with K-Fold CV."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, silhouette_score,
    precision_recall_fscore_support
)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from training.classification.clustering import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt

# Configuration
RANDOM_STATE = 42

def find_optimal_k(X, k_range=range(2, 21), random_state=42, output_dir='output'):
    """Find optimal number of clusters using Elbow and Silhouette methods.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data (already scaled/transformed).
    k_range : range or list
        Range of k values to test.
    random_state : int
        Random seed for reproducibility.
    output_dir : str
        Directory to save plots.

    Returns
    -------
    dict : Dictionary with metrics for each k value
    """
    inertias = []
    silhouette_scores = []
    k_values = list(k_range)

    print("\nFinding optimal k using Elbow and Silhouette methods...")
    print(f"Testing k values from {min(k_values)} to {max(k_values)}")

    for k in k_values:
        print(f"  k={k}...", end=" ")

        # Fit K-Means
        kmeans = KMeans(
            n_clusters=k,
            max_iter=1000,
            n_init=10,
            tol=1e-4,
            random_state=random_state
        )
        cluster_labels = kmeans.fit_predict(X)

        # Compute metrics
        inertia = kmeans.inertia_
        silhouette = silhouette_score(X, cluster_labels, metric='euclidean')

        inertias.append(inertia)
        silhouette_scores.append(silhouette)

        print(f"Inertia={inertia:.2f}, Silhouette={silhouette:.4f}")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Elbow plot
    ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of clusters (k)', fontsize=12)
    ax1.set_ylabel('Inertia (Sum of squared distances)', fontsize=12)
    ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_values)

    # Silhouette plot
    ax2.plot(k_values, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of clusters (k)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(k_values)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)

    # Mark best silhouette
    best_k = k_values[np.argmax(silhouette_scores)]
    best_silhouette = max(silhouette_scores)
    ax2.axvline(x=best_k, color='green', linestyle='--', alpha=0.5,
                label=f'Best k={best_k} (Silhouette={best_silhouette:.4f})')
    ax2.legend()

    plt.tight_layout()
    plot_path = f'{output_dir}/optimal_k_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved optimal k analysis plot to {plot_path}")
    plt.close()

    # Return results
    results = {
        'k_values': k_values,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'best_k_silhouette': best_k,
        'best_silhouette_score': best_silhouette
    }

    print(f"\n{'='*60}")
    print(f"OPTIMAL K RECOMMENDATION:")
    print(f"{'='*60}")
    print(f"Best k (by Silhouette): {best_k}")
    print(f"Silhouette score: {best_silhouette:.4f}")
    print(f"{'='*60}\n")

    return results


class ClusteringClassifier:
    """Clustering-based classifier (K-Means or Agglomerative)."""

    def __init__(self, algorithm='kmeans', n_clusters=5, max_iter=300, n_init=10,
                 tol=1e-4, random_state=None, linkage='ward'):
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = random_state
        self.linkage = linkage
        self.clusterer = None
        self.label_mapping = None

    def fit(self, X, y):
        """Fit clustering algorithm and map clusters to class labels."""
        if self.algorithm == 'kmeans':
            self.clusterer = KMeans(
                n_clusters=self.n_clusters,
                max_iter=self.max_iter,
                n_init=self.n_init,
                tol=self.tol,
                random_state=self.random_state
            )
        elif self.algorithm == 'agglomerative':
            self.clusterer = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage=self.linkage
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        cluster_labels = self.clusterer.fit_predict(X)

        # Map each cluster to the majority class
        self.label_mapping = {}
        self.classes_ = np.unique(y)

        # First pass: assign clusters to their majority class
        for cluster_id in range(self.n_clusters):
            mask = cluster_labels == cluster_id
            if np.sum(mask) > 0:
                cluster_classes = y[mask]
                most_common = Counter(cluster_classes).most_common(1)[0][0]
                self.label_mapping[cluster_id] = most_common

        # Second pass: ensure all classes are represented
        # Find which classes have no clusters assigned
        mapped_classes = set(self.label_mapping.values())
        missing_classes = set(self.classes_) - mapped_classes

        if missing_classes:
            # For each missing class, find the cluster with highest proportion of that class
            for missing_class in missing_classes:
                best_cluster = None
                best_proportion = -1

                for cluster_id in range(self.n_clusters):
                    mask = cluster_labels == cluster_id
                    if np.sum(mask) > 0:
                        cluster_classes = y[mask]
                        proportion = np.sum(cluster_classes == missing_class) / len(cluster_classes)

                        if proportion > best_proportion:
                            best_proportion = proportion
                            best_cluster = cluster_id

                # Reassign this cluster to the missing class
                if best_cluster is not None:
                    self.label_mapping[best_cluster] = missing_class

        # For agglomerative, compute and store centroids for prediction
        if self.algorithm == 'agglomerative':
            self.centroids_ = np.zeros((self.n_clusters, X.shape[1]))
            for cluster_id in range(self.n_clusters):
                mask = cluster_labels == cluster_id
                if np.sum(mask) > 0:
                    self.centroids_[cluster_id] = X[mask].mean(axis=0)

        return self

    def predict(self, X):
        """Predict class labels by assigning to nearest cluster."""
        if self.algorithm == 'kmeans':
            cluster_labels = self.clusterer.predict(X)
        elif self.algorithm == 'agglomerative':
            # Agglomerative doesn't have predict, so we assign to nearest centroid
            # Compute centroids from training data
            if not hasattr(self, 'centroids_'):
                raise RuntimeError("Agglomerative clustering requires centroids. Call fit() first.")
            # Find nearest centroid for each sample
            from scipy.spatial.distance import cdist
            distances = cdist(X, self.centroids_, metric='euclidean')
            cluster_labels = np.argmin(distances, axis=1)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        predictions = np.array([self.label_mapping.get(c, -1) for c in cluster_labels])
        return predictions


def main(data_path, algorithm='kmeans', best_k=None, find_k=False, linkage='ward', agglomerative_subsample=None):
    # Load data
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    if df['Booking Status'].dtype == 'object':
        le = LabelEncoder()
        df['Booking Status'] = le.fit_transform(df['Booking Status'])

    X = df.drop('Booking Status', axis=1)
    y = df['Booking Status']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Downsample to median
    class_counts = Counter(y_train)
    median_count = int(np.median(list(class_counts.values())))
    downsample_strategy = {cls: min(count, median_count) for cls, count in class_counts.items()}
    rus = RandomUnderSampler(sampling_strategy=downsample_strategy, random_state=42)
    X_under, y_under = rus.fit_resample(X_train, y_train)
    target_3 = median_count

    ros = RandomOverSampler(sampling_strategy={3: target_3}, random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_under, y_under)

    print(f"After resampling: {X_resampled.shape}")
    print(f"Class distribution: {Counter(y_resampled)}")

    # Scale first, then PCA (correct order)
    scaler = StandardScaler()
    X_resampled_scaled = scaler.fit_transform(X_resampled)
    X_test_scaled = scaler.transform(X_test)

    # PCA after scaling
    pca = PCA(n_components=0.9, random_state=42)
    X_resampled_pca = pca.fit_transform(X_resampled_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    train_scaled = X_resampled_pca
    test_scaled = X_test_pca

    print(f"PCA reduced to {train_scaled.shape[1]} components explaining {pca.explained_variance_ratio_.sum():.2f} variance")

    y_resampled_array = y_resampled.values if hasattr(y_resampled, 'values') else y_resampled

    # Find optimal k if requested
    if find_k:
        optimal_k_results = find_optimal_k(
            train_scaled,
            k_range=range(2, 21),
            random_state=42,
            output_dir='output'
        )
        if best_k is None:
            best_k = optimal_k_results['best_k_silhouette']
            print(f"Using automatically selected k={best_k}")
    else:
        if best_k is None:
            best_k = 5  # default
            print(f"Using default k={best_k}")

    # Use the selected best_k
    algo_name = algorithm.upper() if algorithm == 'kmeans' else 'Agglomerative'
    print(f"\nTraining {algo_name} with n_clusters={best_k}...")
    if algorithm == 'kmeans':
        print(f"Parameters: n_clusters={best_k}, max_iter=300, n_init=10, tol=1e-4, random_state=42")
    else:
        print(f"Parameters: n_clusters={best_k}, linkage={linkage}")
        if agglomerative_subsample:
            print(f"⚠ Agglomerative: Subsampling to max {agglomerative_subsample} samples per fold for speed")
        print(f"⚠ Note: Agglomerative clustering has O(n³) complexity and may be slow on large datasets")

    # Train final model with K-Fold
    # Reduce folds for agglomerative to speed up training
    n_splits = 3 if algorithm == 'agglomerative' else 11
    skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
    n_classes = len(np.unique(y_resampled_array))

    oof_predictions = np.zeros(len(train_scaled), dtype=int)
    test_predictions_sum = np.zeros((len(test_scaled), n_classes))

    print(f"\nTraining with {n_splits}-fold CV...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_scaled, y_resampled_array)):
        print(f"Fold {fold+1}/{n_splits}...", end=" ")

        # For agglomerative, optionally subsample training data
        if algorithm == 'agglomerative' and agglomerative_subsample is not None:
            subsample_size = min(int(agglomerative_subsample), len(train_idx))
            subsample_idx = np.random.choice(train_idx, size=subsample_size, replace=False)
            train_idx_used = subsample_idx
            print(f"(subsampled {subsample_size} samples) ", end="")
        else:
            train_idx_used = train_idx

        clustering_clf = ClusteringClassifier(
            algorithm=algorithm,
            n_clusters=best_k,
            max_iter=2000,
            n_init=10,
            tol=1e-4,
            random_state=42,
            linkage=linkage
        )
        clustering_clf.fit(train_scaled[train_idx_used], y_resampled_array[train_idx_used])

        # Out-of-fold predictions
        oof_predictions[val_idx] = clustering_clf.predict(train_scaled[val_idx])

        # Test predictions (will be averaged)
        test_pred = clustering_clf.predict(test_scaled)
        for idx, pred_class in enumerate(test_pred):
            if pred_class >= 0:  # valid prediction
                test_predictions_sum[idx, pred_class] += 1

        print("Done")

    # Final test predictions (majority vote)
    test_pred = np.argmax(test_predictions_sum, axis=1)

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {accuracy_score(y_test, test_pred):.4f}")
    print(f"Test F1 (Macro): {f1_score(y_test, test_pred, average='macro'):.4f}")
    print(f"Test F1 (Weighted): {f1_score(y_test, test_pred, average='weighted'):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred))

    # Confusion Matrix
    conf_mat = confusion_matrix(y_test, test_pred)
    print("\nConfusion Matrix (raw counts):")
    print(conf_mat)

    conf_mat_norm = confusion_matrix(y_test, test_pred, normalize='true')
    print("\nConfusion Matrix (normalized):")
    print(np.round(conf_mat_norm, 3))

    # Create output directory if it doesn't exist
    os.makedirs('output/plots', exist_ok=True)

    # Plot confusion matrix
    _, ax = plt.subplots(1, 2, figsize=(12, 5))

    disp1 = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    disp1.plot(ax=ax[0], cmap='Blues', colorbar=False)
    ax[0].set_title("Confusion Matrix (Counts)", fontsize=14, fontweight='bold')

    disp2 = ConfusionMatrixDisplay(confusion_matrix=conf_mat_norm)
    disp2.plot(ax=ax[1], cmap='Blues', colorbar=False)
    ax[1].set_title("Confusion Matrix (Normalized)", fontsize=14, fontweight='bold')

    plt.tight_layout()
    cm_path = f'output/plots/{algorithm}_confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved confusion matrix to {cm_path}")
    plt.close()

    # Plot per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_test, test_pred)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(precision))
    width = 0.25

    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Per-Class Performance Metrics ({algo_name})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Class {i}' for i in range(len(precision))])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.0])

    plt.tight_layout()
    metrics_path = f'output/plots/{algorithm}_class_metrics.png'
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved class metrics plot to {metrics_path}")
    plt.close()

    # Plot prediction distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # True distribution
    true_counts = np.bincount(y_test)
    pred_counts = np.bincount(test_pred, minlength=len(true_counts))

    x = np.arange(len(true_counts))
    ax1.bar(x, true_counts, alpha=0.7, label='True', color='green')
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('True Class Distribution', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Predicted distribution
    ax2.bar(x, pred_counts, alpha=0.7, label='Predicted', color='orange')
    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Predicted Class Distribution', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    dist_path = f'output/plots/{algorithm}_class_distribution.png'
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved class distribution plot to {dist_path}")
    plt.close()

    print(f"\n{'='*60}")
    print("All plots saved to output/plots/")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train clustering-based classifier')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV/Parquet data file')
    parser.add_argument('--algorithm', type=str, default='kmeans', choices=['kmeans', 'agglomerative'],
                        help='Clustering algorithm to use (default: kmeans)')
    parser.add_argument('--best-k', type=int, default=None,
                        help='Number of clusters (if not specified, uses default or auto-detected)')
    parser.add_argument('--find-k', action='store_true',
                        help='Find optimal k using Elbow and Silhouette methods')
    parser.add_argument('--linkage', type=str, default='ward', choices=['ward', 'complete', 'average', 'single'],
                        help='Linkage criterion for agglomerative clustering (default: ward)')
    parser.add_argument('--agglomerative-subsample', type=int, default=5000,
                        help='Maximum samples to use for agglomerative clustering per fold (default: 5000)')

    args = parser.parse_args()
    main(args.data, algorithm=args.algorithm, best_k=args.best_k, find_k=args.find_k,
         linkage=args.linkage, agglomerative_subsample=args.agglomerative_subsample)
