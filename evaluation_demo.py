"""Demonstration of the EvaluationToolkit for both classification and regression."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

from evaluation import EvaluationToolkit


def run_classification_demo(toolkit: EvaluationToolkit) -> None:
    X, y = make_classification(
        n_samples=500,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    history = {
        "loss": list(np.linspace(0.8, 0.2, num=10)),
        "accuracy": list(np.linspace(0.55, 0.92, num=10)),
    }

    metrics = toolkit.generate_classification_dashboard(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        labels=[0, 1],
        history=history,
    )

    print("Classification Metrics")
    print("----------------------")
    print(f"Accuracy: {metrics.accuracy:.3f}")
    print(f"Precision: {metrics.precision:.3f}")
    print(f"Sensitivity (Recall): {metrics.sensitivity:.3f}")
    if metrics.specificity is not None:
        print(f"Specificity: {metrics.specificity:.3f}")
    if metrics.loss is not None:
        print(f"Log-loss: {metrics.loss:.3f}")
    if metrics.roc_auc is not None:
        print(f"ROC AUC: {metrics.roc_auc:.3f}")
    print(f"F1-score: {metrics.f1_score:.3f}")
    print()


def run_regression_demo(toolkit: EvaluationToolkit) -> None:
    X, y = make_regression(
        n_samples=400,
        n_features=5,
        noise=12.0,
        random_state=21,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=21
    )

    reg = LinearRegression()
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    history = {
        "loss": list(np.linspace(1200, 300, num=12)),
        "val_loss": list(np.linspace(1350, 350, num=12)),
    }

    metrics = toolkit.generate_regression_dashboard(
        y_true=y_test,
        y_pred=y_pred,
        history=history,
    )

    print("Regression Metrics")
    print("-------------------")
    print(f"MSE (loss): {metrics.loss:.3f}")
    print(f"RMSE: {metrics.rmse:.3f}")
    print(f"MAE: {metrics.mae:.3f}")
    print(f"R-square: {metrics.r_square:.3f}")
    print()


def main():
    toolkit = EvaluationToolkit()
    run_classification_demo(toolkit)
    run_regression_demo(toolkit)
    plt.show()


if __name__ == "__main__":
    main()
