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
    n_clusters : int, default=3
        Number of clusters to form.
    max_iter : int, default=300
        Maximum number of iterations over the full dataset.
    tol : float, default=1e-4
        Relative tolerance with regards to inertia to declare convergence.
    random_state : int, optional
        Seed controlling centroid initialisation for reproducibility.
    """

    n_clusters: int = 3
    max_iter: int = 300
    tol: float = 1e-4
    random_state: Optional[int] = None

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
        if self.tol < 0:
            raise ValueError("tol must be non-negative")

    # ------------------------------------------------------------------
    def fit(self, X: ArrayLike) -> "KMeansScratch":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        n_samples = X.shape[0]
        if n_samples == 0:
            raise ValueError("X must contain at least one sample")
        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot exceed number of samples")

        rng = np.random.default_rng(self.random_state)
        initial_indices = rng.choice(n_samples, self.n_clusters, replace=False)
        centroids = X[initial_indices].copy()

        for iteration in range(1, self.max_iter + 1):
            distances = self._compute_distances(X, centroids)
            labels = np.argmin(distances, axis=1)

            new_centroids = centroids.copy()
            for cluster_idx in range(self.n_clusters):
                members = X[labels == cluster_idx]
                if members.size == 0:
                    # Re-seed empty clusters to a random point
                    new_centroids[cluster_idx] = X[rng.integers(0, n_samples)]
                else:
                    new_centroids[cluster_idx] = members.mean(axis=0)

            centroid_shift = np.linalg.norm(new_centroids - centroids, axis=1).max()
            centroids = new_centroids

            if centroid_shift <= self.tol:
                self.n_iter_ = iteration
                break
        else:
            distances = self._compute_distances(X, centroids)
            labels = np.argmin(distances, axis=1)
            self.n_iter_ = self.max_iter

        self.cluster_centers_ = centroids
        self.labels_ = labels
        self.inertia_ = float(np.sum((X - centroids[labels]) ** 2))
        self._is_fitted = True
        return self

    def predict(self, X: ArrayLike) -> ArrayLike:
        self._require_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        distances = self._compute_distances(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X: ArrayLike) -> ArrayLike:
        self.fit(X)
        return self.labels_

    def transform(self, X: ArrayLike) -> ArrayLike:
        self._require_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        return self._compute_distances(X, self.cluster_centers_)

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_distances(X: ArrayLike, centroids: ArrayLike) -> ArrayLike:
        diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        return np.linalg.norm(diff, axis=2)

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("KMeansScratch instance is not fitted yet. Call 'fit' first.")


class AgglomerativeClusteringScratch:
    """Bottom-up hierarchical clustering with configurable linkage.

    Parameters
    ----------
    n_clusters : int, default=2
        The number of clusters to find. Must be >= 1.
    linkage : {"average", "single", "complete", "ward"}, default="average"
        The linkage strategy to use when merging clusters.
    """

    def __init__(
        self,
        n_clusters: int = 2,
        linkage: Literal["average", "single", "complete", "ward"] = "average",
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
        self.merge_history_: list[tuple[int, int]] = []
        self.distance_history_: list[float] = []

    # ------------------------------------------------------------------
    def fit(self, X: ArrayLike) -> "AgglomerativeClusteringScratch":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        n_samples = X.shape[0]
        if n_samples == 0:
            raise ValueError("X must contain at least one sample")
        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot exceed number of samples")

        distance_matrix = self._pairwise_distances(X)
        labels = np.arange(n_samples)
        self.merge_history_.clear()
        self.distance_history_.clear()

        while np.unique(labels).size > self.n_clusters:
            np.fill_diagonal(distance_matrix, np.inf)
            row, col = divmod(distance_matrix.argmin(), distance_matrix.shape[1])
            if row == col:
                break

            self.merge_history_.append((int(row), int(col)))
            self.distance_history_.append(float(distance_matrix[row, col]))

            labels[labels == col] = row
            labels[labels > col] -= 1

            distance_matrix = np.delete(distance_matrix, col, axis=0)
            distance_matrix = np.delete(distance_matrix, col, axis=1)

            for idx in range(distance_matrix.shape[0]):
                if idx == row:
                    distance_matrix[row, idx] = np.inf
                    distance_matrix[idx, row] = np.inf
                    continue
                cluster_row = X[labels == row]
                cluster_other = X[labels == idx]
                if cluster_row.size == 0 or cluster_other.size == 0:
                    distance = np.inf
                else:
                    distance = self._linkage_distance(cluster_row, cluster_other)
                distance_matrix[row, idx] = distance_matrix[idx, row] = distance

        self.labels_ = self._relabel_consecutively(labels)
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
