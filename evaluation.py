"""Utility module that gathers common evaluation metrics and plots for
classification and regression models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
	ConfusionMatrixDisplay,
	accuracy_score,
	auc,
	confusion_matrix,
	log_loss,
	precision_recall_fscore_support,
	precision_score,
	recall_score,
	roc_auc_score,
	roc_curve,
	r2_score,
)


@dataclass
class ClassificationMetrics:
	"""Container for the primary classification statistics."""

	loss: Optional[float]
	confusion_matrix: np.ndarray
	accuracy: float
	precision: float
	sensitivity: float
	specificity: Optional[float]
	true_negative_rate: Optional[float]
	f1_score: float
	roc_auc: Optional[float]


@dataclass
class RegressionMetrics:
	"""Container for the primary regression statistics."""

	loss: float
	r_square: float
	rmse: float
	mae: float


class EvaluationToolkit:
	"""Compute metrics and visualise performance for ML models."""

	def __init__(self, positive_label: Optional[int] = 1, average: str = "binary"):
		self.positive_label = positive_label
		self.average = average

	# ------------------------------------------------------------------
	# Classification metrics
	# ------------------------------------------------------------------
	def classification_report(
		self,
		y_true: Sequence[int],
		y_pred: Sequence[int],
		y_proba: Optional[Sequence[float]] = None,
		labels: Optional[Sequence[int]] = None,
	) -> ClassificationMetrics:
		y_true = np.asarray(y_true)
		y_pred = np.asarray(y_pred)
		cm = confusion_matrix(y_true, y_pred, labels=labels)

		accuracy = accuracy_score(y_true, y_pred)

		precision = precision_score(
			y_true,
			y_pred,
			labels=labels,
			average=self.average,
			zero_division=0,
		)

		recall = recall_score(
			y_true,
			y_pred,
			labels=labels,
			average=self.average,
			zero_division=0,
		)

		_, _, f1, _ = precision_recall_fscore_support(
			y_true,
			y_pred,
			labels=labels,
			average=self.average,
			zero_division=0,
		)

		loss_value = None
		roc_auc = None
		specificity = None
		tnr = None

		if y_proba is not None:
			y_proba = np.asarray(y_proba)
			try:
				loss_value = log_loss(y_true, y_proba, labels=labels)
			except ValueError:
				loss_value = None

			try:
				roc_auc = roc_auc_score(y_true, y_proba, average=self.average)
			except ValueError:
				roc_auc = None

		if cm.shape == (2, 2):
			tn, fp, fn, tp = cm.ravel()
			specificity = tn / (tn + fp) if (tn + fp) else None
			tnr = specificity

		return ClassificationMetrics(
			loss=loss_value,
			confusion_matrix=cm,
			accuracy=accuracy,
			precision=precision,
			sensitivity=recall,
			specificity=specificity,
			true_negative_rate=tnr,
			f1_score=f1,
			roc_auc=roc_auc,
		)

	# ------------------------------------------------------------------
	# Regression metrics
	# ------------------------------------------------------------------
	def regression_report(
		self, y_true: Sequence[float], y_pred: Sequence[float]
	) -> RegressionMetrics:
		y_true = np.asarray(y_true)
		y_pred = np.asarray(y_pred)

		residuals = y_true - y_pred
		mse = np.mean(residuals**2)
		rmse = np.sqrt(mse)
		mae = np.mean(np.abs(residuals))
		r_square = r2_score(y_true, y_pred)

		return RegressionMetrics(loss=mse, r_square=r_square, rmse=rmse, mae=mae)

	# ------------------------------------------------------------------
	# Plotting helpers
	# ------------------------------------------------------------------
	@staticmethod
	def plot_confusion_matrix(
		cm: np.ndarray,
		labels: Optional[Sequence[str]] = None,
		normalize: bool = False,
		cmap: str = "Blues",
	) -> plt.Axes:
		plt.figure(figsize=(6, 5))
		display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
		display.plot(values_format=".2f" if normalize else "d", cmap=cmap)
		plt.title("Confusion Matrix")
		plt.tight_layout()
		return display.ax_

	@staticmethod
	def plot_roc_curve(
		y_true: Sequence[int],
		y_score: Sequence[float],
		pos_label: Optional[int] = 1,
		label: str = "ROC Curve",
	) -> plt.Axes:
		fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
		roc_auc = auc(fpr, tpr)
		fig, ax = plt.subplots(figsize=(6, 5))
		ax.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.3f})")
		ax.plot([0, 1], [0, 1], "k--", label="Chance")
		ax.set_xlabel("False Positive Rate")
		ax.set_ylabel("True Positive Rate")
		ax.set_title("ROC Curve")
		ax.legend(loc="lower right")
		ax.grid(True, linestyle="--", alpha=0.4)
		fig.tight_layout()
		return ax

	@staticmethod
	def plot_performance_curves(history: Dict[str, Sequence[float]]) -> List[plt.Axes]:
		"""Plot training history curves such as loss or accuracy.

		Parameters
		----------
		history: dict
			Keys correspond to metric names (e.g., "loss", "val_loss", "accuracy").
			Values must be sequences containing metric values per epoch.
		"""

		if not history:
			raise ValueError("history dictionary is empty")

		metrics = list(history.keys())
		epochs = range(1, len(next(iter(history.values()))) + 1)

		fig, axes = plt.subplots(len(metrics), 1, figsize=(7, 4 * len(metrics)))
		if not isinstance(axes, np.ndarray):
			axes = np.array([axes])

		axes = axes.flatten()

		for ax, metric in zip(axes, metrics):
			ax.plot(epochs, history[metric], marker="o", label=metric)
			ax.set_xlabel("Epoch")
			ax.set_ylabel(metric.replace("_", " ").title())
			ax.set_title(f"{metric.replace('_', ' ').title()} over Epochs")
			ax.grid(True, linestyle="--", alpha=0.4)
			ax.legend()

		fig.tight_layout()
		return list(axes)

	# Convenience wrappers ------------------------------------------------
	def generate_classification_dashboard(
		self,
		y_true: Sequence[int],
		y_pred: Sequence[int],
		y_proba: Optional[Sequence[float]] = None,
		labels: Optional[Sequence[str]] = None,
		history: Optional[Dict[str, Sequence[float]]] = None,
	) -> ClassificationMetrics:
		metrics = self.classification_report(y_true, y_pred, y_proba, labels)
		if metrics.confusion_matrix is not None:
			self.plot_confusion_matrix(metrics.confusion_matrix, labels=labels)

		if y_proba is not None:
			try:
				self.plot_roc_curve(y_true, y_proba, pos_label=self.positive_label)
			except ValueError:
				pass

		if history:
			self.plot_performance_curves(history)

		return metrics

	def generate_regression_dashboard(
		self,
		y_true: Sequence[float],
		y_pred: Sequence[float],
		history: Optional[Dict[str, Sequence[float]]] = None,
	) -> RegressionMetrics:
		metrics = self.regression_report(y_true, y_pred)
		if history:
			self.plot_performance_curves(history)
		return metrics

