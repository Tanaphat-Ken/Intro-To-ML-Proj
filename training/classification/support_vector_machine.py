"""Support Vector Machine implementations from scratch.

This module follows the same structure as the regression implementations under
`training/regression/`, providing clean, educational examples that rely solely
on `numpy` for the optimisation logic. Two variants are included:

* ``LinearSVMClassifier`` – optimised with (sub-)gradient descent on the hinge
  loss for linearly separable or nearly separable data.
* ``KernelSVMClassifier`` – a simple dual-form optimiser that supports
  polynomial and RBF kernels for non-linear decision boundaries.

Both classes expose a familiar API with ``fit`` and ``predict`` methods plus
optional helpers such as ``decision_function`` and plotting routines to track
the optimisation progress.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np


ArrayLike = np.ndarray


class LinearSVMClassifier:
	"""Linear Support Vector Machine optimised with sub-gradient descent.

	Parameters
	----------
	learning_rate : float, default=0.001
		Step size for the gradient updates.
	n_iterations : int, default=1000
		Maximum number of optimisation steps.
	C : float, default=1.0
		Soft-margin penalty parameter. Larger values emphasise margin
		violations more strongly.
	fit_intercept : bool, default=True
		Whether to include a bias term.
	random_state : int, optional
		Seed used when shuffling training data for mini-batching.
	batch_size : int, optional
		When provided, performs mini-batch updates; otherwise uses the full
		dataset for each gradient computation.
	"""

	def __init__(
		self,
		learning_rate: float = 0.001,
		n_iterations: int = 1000,
		C: float = 1.0,
		fit_intercept: bool = True,
		random_state: Optional[int] = None,
		batch_size: Optional[int] = None,
	) -> None:
		self.learning_rate = learning_rate
		self.n_iterations = n_iterations
		self.C = C
		self.fit_intercept = fit_intercept
		self.random_state = random_state
		self.batch_size = batch_size

		self.coef_: Optional[ArrayLike] = None
		self.intercept_: float = 0.0
		self.loss_history: list[float] = []

	# ------------------------------------------------------------------
	def _initialize_parameters(self, n_features: int) -> None:
		rng = np.random.default_rng(self.random_state)
		self.coef_ = rng.normal(loc=0.0, scale=0.01, size=n_features)
		self.intercept_ = 0.0

	def _decision_function(self, X: ArrayLike) -> ArrayLike:
		if self.coef_ is None:
			raise ValueError("Model parameters are not initialized. Call 'fit' first.")
		return X @ self.coef_ + self.intercept_

	def _compute_loss(self, X: ArrayLike, y: ArrayLike) -> float:
		margins = y * self._decision_function(X)
		hinge = np.maximum(0.0, 1.0 - margins)
		return 0.5 * np.dot(self.coef_, self.coef_) + self.C * np.mean(hinge)

	# ------------------------------------------------------------------
	def fit(self, X: ArrayLike, y: ArrayLike) -> "LinearSVMClassifier":
		X = np.asarray(X, dtype=float)
		y = np.asarray(y, dtype=float)
		if X.ndim != 2:
			raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
		if y.ndim != 1:
			raise ValueError("y must be a 1D array")

		n_samples, n_features = X.shape
		if n_samples != y.shape[0]:
			raise ValueError("X and y have incompatible shapes")

		# Convert labels to {-1, 1}
		y_transformed = np.where(y <= 0, -1.0, 1.0)

		self._initialize_parameters(n_features)

		rng = np.random.default_rng(self.random_state)

		for iteration in range(self.n_iterations):
			if self.batch_size is None or self.batch_size >= n_samples:
				batch_indices = np.arange(n_samples)
			else:
				batch_indices = rng.choice(n_samples, self.batch_size, replace=False)

			X_batch = X[batch_indices]
			y_batch = y_transformed[batch_indices]

			margins = y_batch * self._decision_function(X_batch)
			misclassified = margins < 1

			if misclassified.any():
				misclassified_X = X_batch[misclassified]
				misclassified_y = y_batch[misclassified]
				gradient_w = self.coef_ - self.C * (
					misclassified_y @ misclassified_X
				) / misclassified_X.shape[0]
				gradient_b = -self.C * np.mean(misclassified_y) if self.fit_intercept else 0.0
			else:
				gradient_w = self.coef_
				gradient_b = 0.0

			self.coef_ -= self.learning_rate * gradient_w
			if self.fit_intercept:
				self.intercept_ -= self.learning_rate * gradient_b

			loss = self._compute_loss(X, y_transformed)
			self.loss_history.append(loss)

			if iteration > 0 and math.isclose(
				self.loss_history[-2], self.loss_history[-1], rel_tol=1e-6, abs_tol=1e-6
			):
				break

		return self

	# ------------------------------------------------------------------
	def predict(self, X: ArrayLike) -> ArrayLike:
		decision = self._decision_function(np.asarray(X, dtype=float))
		labels = np.where(decision >= 0.0, 1, 0)
		return labels

	def predict_signed(self, X: ArrayLike) -> ArrayLike:
		"""Return signed predictions in {-1, 1}."""

		decision = self._decision_function(np.asarray(X, dtype=float))
		return np.where(decision >= 0.0, 1.0, -1.0)

	def decision_function(self, X: ArrayLike) -> ArrayLike:
		return self._decision_function(np.asarray(X, dtype=float))

	def plot_loss(self, save_path: Optional[str] = None, show: bool = True) -> None:
		if not self.loss_history:
			raise ValueError("No training history available. Train the model first.")
		fig, ax = plt.subplots(figsize=(8, 5))
		ax.plot(self.loss_history, label="Objective")
		ax.set_xlabel("Iteration")
		ax.set_ylabel("Hinge Loss + Regularisation")
		ax.set_title("Linear SVM Optimisation Progress")
		ax.grid(True, linestyle="--", alpha=0.4)
		ax.legend()
		fig.tight_layout()

		if save_path:
			save_path_obj = Path(save_path)
			save_path_obj.parent.mkdir(parents=True, exist_ok=True)
			fig.savefig(save_path_obj, dpi=150, bbox_inches="tight")

		if show:
			plt.show()
		else:
			plt.close(fig)


class OneVsRestLinearSVMClassifier:
	"""Multi-class wrapper that fits one linear SVM per class (one-vs-rest)."""

	def __init__(
		self,
		learning_rate: float = 0.001,
		n_iterations: int = 1000,
		C: float = 1.0,
		fit_intercept: bool = True,
		random_state: Optional[int] = None,
		batch_size: Optional[int] = None,
	) -> None:
		self.learning_rate = learning_rate
		self.n_iterations = n_iterations
		self.C = C
		self.fit_intercept = fit_intercept
		self.random_state = random_state
		self.batch_size = batch_size

		self.classes_: Optional[np.ndarray] = None
		self.classifiers_: list[LinearSVMClassifier] = []
		self.loss_history_: list[list[float]] = []

	def fit(self, X: ArrayLike, y: ArrayLike) -> "OneVsRestLinearSVMClassifier":
		X = np.asarray(X, dtype=float)
		y = np.asarray(y)

		if X.ndim != 2:
			raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
		if y.ndim != 1 or y.shape[0] != X.shape[0]:
			raise ValueError("y must be a 1D array with the same number of samples as X")

		self.classes_ = np.unique(y)
		self.classifiers_ = []
		self.loss_history_ = []

		for idx, cls in enumerate(self.classes_):
			classifier = LinearSVMClassifier(
				learning_rate=self.learning_rate,
				n_iterations=self.n_iterations,
				C=self.C,
				fit_intercept=self.fit_intercept,
				random_state=None if self.random_state is None else self.random_state + idx,
				batch_size=self.batch_size,
			)
			binary_targets = np.where(y == cls, 1.0, 0.0)
			classifier.fit(X, binary_targets)
			self.classifiers_.append(classifier)
			self.loss_history_.append(classifier.loss_history.copy())

		return self

	def decision_function(self, X: ArrayLike) -> ArrayLike:
		if not self.classifiers_:
			raise ValueError("Model is not fitted yet. Call 'fit' first.")
		X = np.asarray(X, dtype=float)
		scores = np.column_stack([
			classifier.decision_function(X) for classifier in self.classifiers_
		])
		return scores

	def predict(self, X: ArrayLike) -> ArrayLike:
		scores = self.decision_function(X)
		indices = np.argmax(scores, axis=1)
		return self.classes_[indices]

	def plot_loss(self, save_path: Optional[str] = None, show: bool = True) -> None:
		if not self.loss_history_:
			raise ValueError("No training history available. Train the model first.")

		fig, ax = plt.subplots(figsize=(8, 5))
		for cls, history in zip(self.classes_, self.loss_history_):
			ax.plot(history, label=str(cls))
		ax.set_xlabel("Iteration")
		ax.set_ylabel("Hinge Loss + Regularisation")
		ax.set_title("One-vs-Rest Linear SVM Training Loss")
		ax.grid(True, linestyle="--", alpha=0.4)
		ax.legend(title="Class")
		fig.tight_layout()

		if save_path:
			save_path_obj = Path(save_path)
			save_path_obj.parent.mkdir(parents=True, exist_ok=True)
			fig.savefig(save_path_obj, dpi=150, bbox_inches="tight")

		if show:
			plt.show()
		else:
			plt.close(fig)


KernelType = Literal["rbf", "poly"]


class KernelSVMClassifier:
	"""Kernelised SVM trained in the dual using projected gradient ascent.

	Parameters
	----------
	kernel : {"rbf", "poly"}, default="rbf"
		Kernel type to use. The RBF kernel behaves similarly to a Gaussian
		radial basis function, while the polynomial kernel produces curved
		decision boundaries of configurable degree.
	C : float, default=1.0
		Soft-margin penalty parameter.
	degree : int, default=3
		Degree of the polynomial kernel (only used when ``kernel="poly"``).
	gamma : float, optional
		Kernel coefficient for RBF. If omitted, defaults to ``1 / n_features``.
	coef0 : float, default=1.0
		Independent term added to the polynomial kernel.
	learning_rate : float, default=0.001
		Step size for the projected gradient ascent on the dual variables.
	n_iterations : int, default=1000
		Maximum number of optimisation iterations.
	tolerance : float, default=1e-4
		Threshold for early stopping on the dual objective improvement.
	"""

	def __init__(
		self,
		kernel: KernelType = "rbf",
		C: float = 1.0,
		degree: int = 3,
		gamma: Optional[float] = None,
		coef0: float = 1.0,
		learning_rate: float = 0.001,
		n_iterations: int = 1000,
		tolerance: float = 1e-4,
	) -> None:
		if kernel not in {"rbf", "poly"}:
			raise ValueError("kernel must be either 'rbf' or 'poly'")

		self.kernel = kernel
		self.C = C
		self.degree = degree
		self.gamma = gamma
		self.coef0 = coef0
		self.learning_rate = learning_rate
		self.n_iterations = n_iterations
		self.tolerance = tolerance

		self.alpha_: Optional[ArrayLike] = None
		self.support_vectors_: Optional[ArrayLike] = None
		self.support_vector_labels_: Optional[ArrayLike] = None
		self.bias_: float = 0.0
		self.objective_history: list[float] = []

	# ------------------------------------------------------------------
	def _compute_kernel(self, X: ArrayLike, Z: ArrayLike) -> ArrayLike:
		if self.kernel == "rbf":
			gamma = self.gamma or 1.0 / X.shape[1]
			diff = X[:, np.newaxis, :] - Z[np.newaxis, :, :]
			return np.exp(-gamma * np.sum(diff**2, axis=2))

		# Polynomial kernel
		return (self.coef0 + X @ Z.T) ** self.degree

	def _dual_objective(self, kernel_matrix: ArrayLike, y: ArrayLike, alpha: ArrayLike) -> float:
		return np.sum(alpha) - 0.5 * np.sum(
			(alpha[:, None] * alpha[None, :]) * (y[:, None] * y[None, :]) * kernel_matrix
		)

	# ------------------------------------------------------------------
	def fit(self, X: ArrayLike, y: ArrayLike) -> "KernelSVMClassifier":
		X = np.asarray(X, dtype=float)
		y = np.asarray(y, dtype=float)
		if X.ndim != 2:
			raise ValueError("X must be a 2D array")
		if y.ndim != 1:
			raise ValueError("y must be a 1D array")

		n_samples = X.shape[0]
		if n_samples != y.shape[0]:
			raise ValueError("X and y have incompatible shapes")

		labels = np.where(y <= 0, -1.0, 1.0)
		kernel_matrix = self._compute_kernel(X, X)
		scaled_kernel = (labels[:, None] * labels[None, :]) * kernel_matrix

		alpha = np.zeros(n_samples)
		previous_objective = -np.inf

		for iteration in range(self.n_iterations):
			gradient = 1.0 - scaled_kernel @ alpha
			alpha += self.learning_rate * gradient
			alpha = np.clip(alpha, 0.0, self.C)

			objective = self._dual_objective(kernel_matrix, labels, alpha)
			self.objective_history.append(objective)

			if iteration > 0 and abs(objective - previous_objective) < self.tolerance:
				break
			previous_objective = objective

		support_mask = alpha > 1e-6
		self.alpha_ = alpha[support_mask]
		self.support_vectors_ = X[support_mask]
		self.support_vector_labels_ = labels[support_mask]

		if self.alpha_.size == 0:
			raise RuntimeError("Training failed to find support vectors. Try adjusting parameters.")

		decision = (self.alpha_ * self.support_vector_labels_) @ self._compute_kernel(
			self.support_vectors_, self.support_vectors_
		)

		# Compute bias using the KKT condition on support vectors with 0 < alpha < C
		sv_margin_mask = (alpha[support_mask] > 1e-6) & (alpha[support_mask] < self.C - 1e-6)
		if np.any(sv_margin_mask):
			sv_indices = np.where(sv_margin_mask)[0]
		else:
			sv_indices = np.arange(self.alpha_.size)

		bias_terms = []
		for idx in sv_indices:
			K_sv = self._compute_kernel(self.support_vectors_, self.support_vectors_[idx : idx + 1])
			decision_value = np.sum(
				self.alpha_ * self.support_vector_labels_ * K_sv[:, 0]
			)
			bias_terms.append(self.support_vector_labels_[idx] - decision_value)

		self.bias_ = float(np.mean(bias_terms))
		return self

	# ------------------------------------------------------------------
	def decision_function(self, X: ArrayLike) -> ArrayLike:
		if self.alpha_ is None or self.support_vectors_ is None:
			raise ValueError("Model is not fitted yet. Call 'fit' first.")
		X = np.asarray(X, dtype=float)
		K = self._compute_kernel(X, self.support_vectors_)
		return K @ (self.alpha_ * self.support_vector_labels_) + self.bias_

	def predict(self, X: ArrayLike) -> ArrayLike:
		scores = self.decision_function(X)
		return np.where(scores >= 0.0, 1, 0)

	def predict_signed(self, X: ArrayLike) -> ArrayLike:
		scores = self.decision_function(X)
		return np.where(scores >= 0.0, 1.0, -1.0)

	def plot_objective(self, save_path: Optional[str] = None, show: bool = True) -> None:
		if not self.objective_history:
			raise ValueError("No optimisation history recorded. Call 'fit' first.")
		fig, ax = plt.subplots(figsize=(8, 5))
		ax.plot(self.objective_history, label="Dual Objective")
		ax.set_xlabel("Iteration")
		ax.set_ylabel("Objective Value")
		ax.set_title("Kernel SVM Dual Optimisation")
		ax.grid(True, linestyle="--", alpha=0.4)
		ax.legend()
		fig.tight_layout()

		if save_path:
			save_path_obj = Path(save_path)
			save_path_obj.parent.mkdir(parents=True, exist_ok=True)
			fig.savefig(save_path_obj, dpi=150, bbox_inches="tight")

		if show:
			plt.show()
		else:
			plt.close(fig)


class SVM_Linear_Scratch(LinearSVMClassifier):
	"""Compatiblity wrapper that mirrors the notebook-friendly API.

	This class keeps the public surface that appears in
	``W5-1_Support_Vector_Machine_(SVM).ipynb`` while delegating the heavy
	lifting to :class:`LinearSVMClassifier`. It means you can keep using the
	``fit`` return values and helper methods expected in the lecture material,
	but benefit from the vectorised and well-tested implementation above.
	"""

	def __init__(
		self,
		C: float = 1.0,
		batch_size: int | None = 100,
		learning_rate: float = 0.001,
		iterations: int = 1000,
		fit_intercept: bool = True,
		random_state: int | None = None,
	) -> None:
		super().__init__(
			learning_rate=learning_rate,
			n_iterations=iterations,
			C=C,
			fit_intercept=fit_intercept,
			random_state=random_state,
			batch_size=batch_size,
		)

	def hingeloss(
		self, w: ArrayLike, b: float, X: ArrayLike, y: ArrayLike
	) -> float:
		"""Match the helper used in the lecture notebook for monitoring loss."""

		labels = np.where(np.asarray(y) <= 0, -1.0, 1.0)
		margins = labels * (np.asarray(X) @ w + b)
		hinge = np.maximum(0.0, 1.0 - margins)
		return 0.5 * float(np.dot(w, w)) + self.C * float(np.mean(hinge))

	def fit(self, X: ArrayLike, y: ArrayLike):
		super().fit(X, y)
		return self.coef_, self.intercept_, self.loss_history


class SVM_Non_Linear_Scratch(KernelSVMClassifier):
	"""Notebook-style wrapper around :class:`KernelSVMClassifier`.

	Parameters mirror the lecturer-provided example while deferring to the
	efficient dual optimiser implemented above. When using the RBF kernel, the
	``sigma`` argument is translated to the equivalent ``gamma``.
	"""

	def __init__(
		self,
		kernel: Literal["poly", "rbf"] = "poly",
		C: float = 1.0,
		degree: int = 2,
		const: float = 1.0,
		sigma: float = 0.1,
		iterations: int = 1000,
		learning_rate: float = 0.001,
		tolerance: float = 1e-4,
	) -> None:
		gamma = None
		if kernel == "rbf" and sigma > 0.0:
			gamma = 1.0 / (sigma**2)
		super().__init__(
			kernel=kernel,
			C=C,
			degree=degree,
			gamma=gamma,
			coef0=const,
			learning_rate=learning_rate,
			n_iterations=iterations,
			tolerance=tolerance,
		)
		self._sigma = sigma
		self._const = const

	def polynomial_kernel(self, X: ArrayLike, Z: ArrayLike) -> ArrayLike:
		return (self._const + np.asarray(X) @ np.asarray(Z).T) ** self.degree

	def gaussian_kernel(self, X: ArrayLike, Z: ArrayLike) -> ArrayLike:
		sigma = self._sigma if self._sigma > 0.0 else 1.0
		diff = np.asarray(X)[:, np.newaxis, :] - np.asarray(Z)[np.newaxis, :, :]
		return np.exp(-(1.0 / (sigma**2)) * np.sum(diff**2, axis=2))

	def fit(self, X: ArrayLike, y: ArrayLike):
		if self.kernel == "poly":
			self.gamma = None
		elif self.kernel == "rbf" and self._sigma > 0.0:
			self.gamma = 1.0 / (self._sigma**2)
		super().fit(X, y)
		return self.alpha_, self.bias_, self.objective_history

