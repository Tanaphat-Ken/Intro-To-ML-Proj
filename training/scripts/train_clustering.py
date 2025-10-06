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
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, silhouette_score
from training.classification.clustering import KMeansScratch
import matplotlib.pyplot as plt

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
        kmeans = KMeansScratch(
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


class KMeansClassifier:
    """K-Means clustering used for classification via nearest centroid."""

    def __init__(self, n_clusters=5, max_iter=300, n_init=10, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = random_state
        self.kmeans = None
        self.label_mapping = None

    def fit(self, X, y):
        """Fit K-Means and map clusters to class labels."""
        self.kmeans = KMeansScratch(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            n_init=self.n_init,
            tol=self.tol,
            random_state=self.random_state
        )
        cluster_labels = self.kmeans.fit_predict(X)

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

        return self

    def predict(self, X):
        """Predict class labels by assigning to nearest cluster."""
        cluster_labels = self.kmeans.predict(X)
        predictions = np.array([self.label_mapping.get(c, -1) for c in cluster_labels])
        return predictions


def main(data_path, best_k=None, find_k=False):
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
    print(f"\nTraining K-Means with n_clusters={best_k}...")
    print(f"Parameters: n_clusters={best_k}, max_iter=300, n_init=10, tol=1e-4, random_state=42")

    # Train final model with K-Fold
    n_splits = 11
    skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
    n_classes = len(np.unique(y_resampled_array))

    oof_predictions = np.zeros(len(train_scaled), dtype=int)
    test_predictions_sum = np.zeros((len(test_scaled), n_classes))

    print(f"\nTraining with {n_splits}-fold CV...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_scaled, y_resampled_array)):
        print(f"Fold {fold+1}/{n_splits}...", end=" ")

        kmeans_clf = KMeansClassifier(
            n_clusters=best_k,
            max_iter=2000,
            n_init=10,
            tol=1e-4,
            random_state=42
        )
        kmeans_clf.fit(train_scaled[train_idx], y_resampled_array[train_idx])

        # Out-of-fold predictions
        oof_predictions[val_idx] = kmeans_clf.predict(train_scaled[val_idx])

        # Test predictions (will be averaged)
        test_pred = kmeans_clf.predict(test_scaled)
        for idx, pred_class in enumerate(test_pred):
            if pred_class >= 0:  # valid prediction
                test_predictions_sum[idx, pred_class] += 1

        print("Done")

    # Final test predictions (majority vote)
    test_pred = np.argmax(test_predictions_sum, axis=1)

    print(f"\nTest Accuracy: {accuracy_score(y_test, test_pred):.4f}")
    print(f"Test F1 (Macro): {f1_score(y_test, test_pred, average='macro'):.4f}")
    print(f"Test F1 (Weighted): {f1_score(y_test, test_pred, average='weighted'):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred))

    conf_mat = confusion_matrix(y_test, test_pred)
    print("\nConfusion Matrix (raw counts):")
    print(conf_mat)

    # Optional: normalized version (percentages)
    conf_mat_norm = confusion_matrix(y_test, test_pred, normalize='true')
    print("\nConfusion Matrix (normalized):")
    print(np.round(conf_mat_norm, 3))

    # Plot confusion matrix
    _, ax = plt.subplots(1, 2, figsize=(12, 5))

    disp1 = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    disp1.plot(ax=ax[0], cmap='Blues', colorbar=False)
    ax[0].set_title("Confusion Matrix (Counts)")

    disp2 = ConfusionMatrixDisplay(confusion_matrix=conf_mat_norm)
    disp2.plot(ax=ax[1], cmap='Blues', colorbar=False)
    ax[1].set_title("Confusion Matrix (Normalized)")

    plt.tight_layout()
    plt.savefig("output/confusion_matrix.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved confusion matrix to output/confusion_matrix.png")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train K-Means clustering classifier')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV/Parquet data file')
    parser.add_argument('--best-k', type=int, default=None, help='Number of clusters (if not specified, uses default or auto-detected)')
    parser.add_argument('--find-k', action='store_true', help='Find optimal k using Elbow and Silhouette methods')

    args = parser.parse_args()
    main(args.data, best_k=args.best_k, find_k=args.find_k)
