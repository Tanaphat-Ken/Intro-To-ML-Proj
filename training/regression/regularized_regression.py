import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

class RidgeRegression:
    """
    Ridge Regression (L2 Regularization): β = (X^T X + αI)^(-1) X^T y

    Parameters:
    -----------
    alpha : float, default=1.0
        Regularization strength (must be positive). Benchmark default: 1.0
    learning_rate : float, default=0.01
        Learning rate for gradient descent. Benchmark default: 0.01
    n_iterations : int, default=1000
        Maximum training iterations. Benchmark default: 1000
    tolerance : float, default=1e-4
        Convergence threshold. Benchmark default: 1e-4
    fit_intercept : bool, default=True
        Whether to calculate intercept
    """

    def __init__(self, alpha=1.0, learning_rate=0.01, n_iterations=1000, tolerance=1e-4, fit_intercept=True):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.fit_intercept = fit_intercept
        self.coefficients = None
        self.intercept = None
        self.loss_history = []

    def fit(self, X, y):
        """
        Fit Ridge regression model.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        # Initialize weights
        self.coefficients = np.zeros(n_features)
        self.intercept = 0.0

        for iteration in range(self.n_iterations):
            y_predicted = X @ self.coefficients + self.intercept

            # Calculate loss with L2 penalty
            mse_loss = (1 / (2 * n_samples)) * np.sum((y - y_predicted) ** 2)
            l2_penalty = (self.alpha / 2) * np.sum(self.coefficients ** 2)
            total_loss = mse_loss + l2_penalty
            self.loss_history.append(total_loss)

            # Early stopping with relative tolerance
            if iteration > 0:
                prev_loss = self.loss_history[-2]
                if prev_loss > 0:
                    relative_change = abs(prev_loss - total_loss) / prev_loss
                    if relative_change < self.tolerance:
                        print(f"Ridge converged at iteration {iteration+1} (relative change: {relative_change:.6f})")
                        break

            # Calculate gradients
            gradient_coefficients = (-1 / n_samples) * X.T @ (y - y_predicted) + self.alpha * self.coefficients
            gradient_intercept = (-1 / n_samples) * np.sum(y - y_predicted)

            # Update parameters
            self.coefficients -= self.learning_rate * gradient_coefficients
            self.intercept -= self.learning_rate * gradient_intercept

        return self.coefficients, self.intercept, self.loss_history

    def predict(self, X):
        """
        Predict using Ridge regression model.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input samples

        Returns:
        --------
        array, shape (n_samples,)
            Predicted values
        """
        X = np.array(X)
        return X @ self.coefficients + self.intercept

    def plot_loss(self, save_path=None):
        """Plot the loss curve during training."""
        plt.figure(figsize=(8, 5))
        plt.plot(range(len(self.loss_history)), self.loss_history, label='Ridge Loss (MSE + L2)')
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Loss Curve: Ridge Regression")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.show()

    def save_weights(self, filepath):
        """Save model weights to file."""
        weights = {k: v for k, v in self.__dict__.items()}
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(weights, f)
        print(f"Model weights saved to {filepath}")

    def load_weights(self, filepath):
        """Load model weights from file."""
        with open(filepath, 'rb') as f:
            weights = pickle.load(f)
        for k, v in weights.items():
            setattr(self, k, v)
        print(f"Model weights loaded from {filepath}")


class LassoRegression:
    """
    Lasso Regression (L1 Regularization) with soft-thresholding.

    Parameters:
    -----------
    alpha : float, default=1.0
        Regularization strength (must be positive). Benchmark default: 1.0
    learning_rate : float, default=0.01
        Learning rate for gradient descent. Benchmark default: 0.01
    n_iterations : int, default=1000
        Maximum training iterations. Benchmark default: 1000
    tolerance : float, default=1e-4
        Convergence threshold. Benchmark default: 1e-4
    fit_intercept : bool, default=True
        Whether to calculate intercept
    """

    def __init__(self, alpha=1.0, learning_rate=0.01, n_iterations=1000, tolerance=1e-4, fit_intercept=True):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.fit_intercept = fit_intercept
        self.coefficients = None
        self.intercept = None
        self.loss_history = []

    def fit(self, X, y):
        """
        Fit Lasso regression model with L1 regularization.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        # Initialize weights
        self.coefficients = np.zeros(n_features)
        self.intercept = 0.0

        for iteration in range(self.n_iterations):
            y_predicted = X @ self.coefficients + self.intercept

            # Calculate loss with L1 penalty
            mse_loss = (1 / (2 * n_samples)) * np.sum((y - y_predicted) ** 2)
            l1_penalty = self.alpha * np.sum(np.abs(self.coefficients))
            total_loss = mse_loss + l1_penalty
            self.loss_history.append(total_loss)

            # Early stopping with relative tolerance
            if iteration > 0:
                prev_loss = self.loss_history[-2]
                if prev_loss > 0:
                    relative_change = abs(prev_loss - total_loss) / prev_loss
                    if relative_change < self.tolerance:
                        print(f"Lasso converged at iteration {iteration+1} (relative change: {relative_change:.6f})")
                        break

            # Calculate gradients with L1 penalty (using sign function)
            gradient_coefficients = (-1 / n_samples) * X.T @ (y - y_predicted) + self.alpha * np.sign(self.coefficients)
            gradient_intercept = (-1 / n_samples) * np.sum(y - y_predicted)

            # Update parameters
            self.coefficients -= self.learning_rate * gradient_coefficients
            self.intercept -= self.learning_rate * gradient_intercept

        return self.coefficients, self.intercept, self.loss_history

    def predict(self, X):
        """
        Predict using Lasso regression model.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input samples

        Returns:
        --------
        array, shape (n_samples,)
            Predicted values
        """
        X = np.array(X)
        return X @ self.coefficients + self.intercept

    def get_selected_features(self, feature_names=None, threshold=1e-5):
        """
        Get features selected by Lasso (non-zero coefficients).

        Parameters:
        -----------
        feature_names : list, optional
            Names of features
        threshold : float, default=1e-5
            Threshold below which coefficients are considered zero

        Returns:
        --------
        dict
            Dictionary mapping feature names/indices to their coefficients
        """
        selected_indices = np.where(np.abs(self.coefficients) > threshold)[0]

        if feature_names is not None:
            return {feature_names[idx]: self.coefficients[idx] for idx in selected_indices}
        else:
            return {f"feature_{idx}": self.coefficients[idx] for idx in selected_indices}

    def plot_loss(self, save_path=None):
        """Plot the loss curve during training."""
        plt.figure(figsize=(8, 5))
        plt.plot(range(len(self.loss_history)), self.loss_history, label='Lasso Loss (MSE + L1)')
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Loss Curve: Lasso Regression")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.show()

    def save_weights(self, filepath):
        """Save model weights to file."""
        weights = {k: v for k, v in self.__dict__.items()}
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(weights, f)
        print(f"Model weights saved to {filepath}")

    def load_weights(self, filepath):
        """Load model weights from file."""
        with open(filepath, 'rb') as f:
            weights = pickle.load(f)
        for k, v in weights.items():
            setattr(self, k, v)
        print(f"Model weights loaded from {filepath}")
