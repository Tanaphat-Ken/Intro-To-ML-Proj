"""Clustering algorithms implemented from scratch.

This module mirrors the style used across the training package by exposing
clean, NumPy-driven implementations of unsupervised clustering techniques.
Two algorithms are provided:

* ``KMeansScratch`` – classic Lloyd-style K-Means clustering with support for
  inertia tracking and transform helpers.
* ``AgglomerativeClusteringScratch`` – bottom-up hierarchical clustering with
  configurable linkage criteria.

Both classes adopt a scikit-learn-like API offering ``fit`` and
``fit_predict`` methods plus convenience accessors for cluster labels and
centroids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

ArrayLike = np.ndarray


@dataclass(slots=True)
class KMeansScratch:
    """K-Means clustering using Lloyd's algorithm.

    Parameters
    ----------
    n_clusters : int, default=5
        Number of clusters to form.
    max_iter : int, default=300
        Maximum number of iterations over the full dataset.
    n_init : int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.
    tol : float, default=1e-4
        Relative tolerance with regards to inertia to declare convergence.
        Convergence is declared when: |prev_inertia - inertia| <= tol * max(|prev_inertia|, 1e-12).
    random_state : int, optional
        Seed controlling centroid initialisation for reproducibility.
    init : str, default="k-means++"
        Method for initialization: "k-means++" (smart initialization) or "random".
    """

    n_clusters: int = 5
    max_iter: int = 300
    n_init: int = 10
    tol: float = 1e-4
    random_state: Optional[int] = None
    init: str = "k-means++"  # "k-means++" or "random"

    cluster_centers_: Optional[ArrayLike] = field(init=False, default=None, repr=False)
    labels_: Optional[ArrayLike] = field(init=False, default=None, repr=False)
    inertia_: Optional[float] = field(init=False, default=None, repr=False)
    n_iter_: int = field(init=False, default=0)
    _is_fitted: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        if self.n_init <= 0:
            raise ValueError("n_init must be a positive integer")
        if self.tol < 0:
            raise ValueError("tol must be non-negative")

    # ------------------------------------------------------------------
    def fit(self, X: ArrayLike) -> "KMeansScratch":
        """Fit K-Means clustering with multiple random initializations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : KMeansScratch
            Fitted estimator.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        n_samples = X.shape[0]
        if n_samples == 0:
            raise ValueError("X must contain at least one sample")
        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot exceed number of samples")

        # Run K-Means n_init times and keep the best result
        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        best_n_iter = 0

        rng = np.random.default_rng(self.random_state)

        for init_idx in range(self.n_init):
            # Generate unique seed for each initialization
            seed = int(rng.integers(0, 2**31 - 1))
            centroids, labels, inertia, n_iter = self._fit_single(X, seed)

            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_n_iter = n_iter

        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        self._is_fitted = True
        return self

    def _fit_single(self, X: ArrayLike, seed: Optional[int]) -> tuple:
        """Single run of K-Means algorithm.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        seed : int or None
            Random seed for this initialization.

        Returns
        -------
        centroids : array of shape (n_clusters, n_features)
            Final cluster centers.
        labels : array of shape (n_samples,)
            Cluster assignments.
        inertia : float
            Sum of squared distances to nearest cluster center.
        n_iter : int
            Number of iterations run.
        """
        n_samples = X.shape[0]
        rng = np.random.default_rng(seed)

        # Initialize centroids
        if self.init == "k-means++":
            centroids = self._kmeans_plus_plus_init(X, rng)
        else:  # random
            initial_indices = rng.choice(n_samples, self.n_clusters, replace=False)
            centroids = X[initial_indices].copy()

        prev_inertia = np.inf

        for iteration in range(1, self.max_iter + 1):
            # Assign samples to nearest centroid (using squared distances)
            d2 = self._compute_distances_squared(X, centroids)
            labels = np.argmin(d2, axis=1)

            # Compute inertia
            inertia = float(np.sum(d2[np.arange(n_samples), labels]))

            # Check convergence based on inertia change
            if abs(prev_inertia - inertia) <= self.tol * max(abs(prev_inertia), 1e-12):
                n_iter = iteration
                break
            prev_inertia = inertia

            # Update centroids
            new_centroids = centroids.copy()
            for cluster_idx in range(self.n_clusters):
                members = X[labels == cluster_idx]
                if members.size == 0:
                    # Re-seed empty clusters to a random point
                    new_centroids[cluster_idx] = X[rng.integers(0, n_samples)]
                else:
                    new_centroids[cluster_idx] = members.mean(axis=0)

            centroids = new_centroids
        else:
            # Recalculate labels and inertia if max_iter reached
            d2 = self._compute_distances_squared(X, centroids)
            labels = np.argmin(d2, axis=1)
            inertia = float(np.sum(d2[np.arange(n_samples), labels]))
            n_iter = self.max_iter

        return centroids, labels, inertia, n_iter

    def _kmeans_plus_plus_init(self, X: ArrayLike, rng) -> ArrayLike:
        """Initialize centroids using k-means++ algorithm.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        rng : numpy random generator
            Random number generator.

        Returns
        -------
        centroids : array of shape (n_clusters, n_features)
            Initial cluster centers.
        """
        n_samples, n_features = X.shape
        centroids = np.empty((self.n_clusters, n_features), dtype=X.dtype)

        # Pick first center randomly
        idx0 = int(rng.integers(0, n_samples))
        centroids[0] = X[idx0]

        # Pick remaining centers
        for j in range(1, self.n_clusters):
            d2 = np.min(self._compute_distances_squared(X, centroids[:j]), axis=1)
            # Clip to ensure non-negative (handle numerical precision issues)
            d2 = np.maximum(d2, 0.0)
            d2_sum = d2.sum()

            if d2_sum < 1e-12:
                # All points are extremely close to existing centroids
                # Just pick a random point
                idx = int(rng.integers(0, n_samples))
            else:
                probs = d2 / d2_sum
                # Ensure probabilities sum to 1.0 and are non-negative
                probs = np.maximum(probs, 0.0)
                probs = probs / probs.sum()
                idx = int(rng.choice(n_samples, p=probs))

            centroids[j] = X[idx]

        return centroids

    def predict(self, X: ArrayLike) -> ArrayLike:
        self._require_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        d2 = self._compute_distances_squared(X, self.cluster_centers_)
        return np.argmin(d2, axis=1)

    def fit_predict(self, X: ArrayLike) -> ArrayLike:
        self.fit(X)
        return self.labels_

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Transform X to cluster-distance space (squared distances).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : array of shape (n_samples, n_clusters)
            Squared distances to cluster centers.
        """
        self._require_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        return self._compute_distances_squared(X, self.cluster_centers_)

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_distances_squared(X: ArrayLike, centroids: ArrayLike) -> ArrayLike:
        """Compute squared Euclidean distances efficiently.

        Using the formula: ||x - c||^2 = ||x||^2 - 2*x·c + ||c||^2

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        centroids : array of shape (n_clusters, n_features)

        Returns
        -------
        distances_squared : array of shape (n_samples, n_clusters)
        """
        X2 = np.sum(X**2, axis=1, keepdims=True)  # (n, 1)
        C2 = np.sum(centroids**2, axis=1, keepdims=True).T  # (1, k)
        XC = X @ centroids.T  # (n, k)
        return X2 - 2 * XC + C2

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("KMeansScratch instance is not fitted yet. Call 'fit' first.")


class AgglomerativeClusteringScratch:
    """Bottom-up hierarchical clustering with configurable linkage.

    Parameters
    ----------
    n_clusters : int, default=5
        The number of clusters to find. Must be >= 1.
    linkage : {"ward", "complete", "average", "single"}, default="ward"
        The linkage strategy to use when merging clusters.
        - 'ward': minimizes variance within clusters (recommended)
        - 'complete': maximum distance between cluster pairs
        - 'average': average distance between cluster pairs
        - 'single': minimum distance between cluster pairs
    """

    def __init__(
        self,
        n_clusters: int = 5,
        linkage: Literal["average", "single", "complete", "ward"] = "ward",
    ) -> None:
        if n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer")
        self.n_clusters = n_clusters

        linkage = linkage.lower()
        if linkage == "average":
            self._linkage_distance = self._average_linkage
        elif linkage == "single":
            self._linkage_distance = self._single_linkage
        elif linkage == "complete":
            self._linkage_distance = self._complete_linkage
        elif linkage == "ward":
            self._linkage_distance = self._ward_linkage
        else:
            raise ValueError("Unsupported linkage. Choose from 'average', 'single', 'complete', 'ward'.")
        self.linkage = linkage

        self.labels_: Optional[ArrayLike] = None
        self.n_clusters_: Optional[int] = None
        self.merge_history_: list[tuple[int, int]] = []
        self.distance_history_: list[float] = []

    # ------------------------------------------------------------------
    def fit(self, X: ArrayLike) -> "AgglomerativeClusteringScratch":
        """Fit agglomerative clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : AgglomerativeClusteringScratch
            Fitted estimator.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        n_samples = X.shape[0]
        if n_samples == 0:
            raise ValueError("X must contain at least one sample")
        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot exceed number of samples")

        # Compute initial pairwise distance matrix
        distance_matrix = self._pairwise_distances(X)
        labels = np.arange(n_samples)
        self.merge_history_.clear()
        self.distance_history_.clear()

        # Iteratively merge closest clusters
        while np.unique(labels).size > self.n_clusters:
            # Ensure diagonal is inf before finding minimum
            np.fill_diagonal(distance_matrix, np.inf)

            row, col = divmod(distance_matrix.argmin(), distance_matrix.shape[1])
            if row == col:
                break

            self.merge_history_.append((int(row), int(col)))
            self.distance_history_.append(float(distance_matrix[row, col]))

            # Merge cluster col into cluster row
            labels[labels == col] = row
            labels[labels > col] -= 1

            # Update distance matrix
            distance_matrix = np.delete(distance_matrix, col, axis=0)
            distance_matrix = np.delete(distance_matrix, col, axis=1)

            # Recompute distances for merged cluster
            for idx in range(distance_matrix.shape[0]):
                if idx == row:
                    continue
                cluster_row = X[labels == row]
                cluster_other = X[labels == idx]
                if cluster_row.size == 0 or cluster_other.size == 0:
                    distance = np.inf
                else:
                    distance = self._linkage_distance(cluster_row, cluster_other)
                distance_matrix[row, idx] = distance_matrix[idx, row] = distance

            # Explicitly set diagonal to inf after update
            np.fill_diagonal(distance_matrix, np.inf)

        self.labels_ = self._relabel_consecutively(labels)
        self.n_clusters_ = len(np.unique(self.labels_))
        return self

    def fit_predict(self, X: ArrayLike) -> ArrayLike:
        self.fit(X)
        return self.labels_

    # ------------------------------------------------------------------
    @staticmethod
    def _pairwise_distances(X: ArrayLike) -> ArrayLike:
        n_samples = X.shape[0]
        distance_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)
        for i in range(n_samples):
            diff = X[i + 1 :] - X[i]
            norms = np.linalg.norm(diff, axis=1)
            if norms.size:
                norms = norms.astype(np.float32, copy=False)
                distance_matrix[i, i + 1 :] = norms
                distance_matrix[i + 1 :, i] = norms
        return distance_matrix

    @staticmethod
    def _average_linkage(cluster_a: ArrayLike, cluster_b: ArrayLike) -> float:
        distances = np.linalg.norm(cluster_a[:, None, :] - cluster_b[None, :, :], axis=2)
        return float(distances.mean())

    @staticmethod
    def _single_linkage(cluster_a: ArrayLike, cluster_b: ArrayLike) -> float:
        distances = np.linalg.norm(cluster_a[:, None, :] - cluster_b[None, :, :], axis=2)
        return float(distances.min())

    @staticmethod
    def _complete_linkage(cluster_a: ArrayLike, cluster_b: ArrayLike) -> float:
        distances = np.linalg.norm(cluster_a[:, None, :] - cluster_b[None, :, :], axis=2)
        return float(distances.max())

    @staticmethod
    def _ward_linkage(cluster_a: ArrayLike, cluster_b: ArrayLike) -> float:
        n_a = cluster_a.shape[0]
        n_b = cluster_b.shape[0]
        centroid_a = cluster_a.mean(axis=0)
        centroid_b = cluster_b.mean(axis=0)
        diff = centroid_a - centroid_b
        return float((n_a * n_b) / (n_a + n_b) * np.dot(diff, diff))

    @staticmethod
    def _relabel_consecutively(labels: ArrayLike) -> ArrayLike:
        unique_labels = np.unique(labels)
        remap = {old: new for new, old in enumerate(unique_labels)}
        return np.vectorize(remap.get)(labels)
