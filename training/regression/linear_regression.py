import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

class LinearRegression:
    """
    Linear Regression using Normal Equation: β = (X^T X)^(-1) X^T y

    Parameters:
    -----------
    fit_intercept : bool, default=True
        Whether to calculate intercept for the model
    """

    def __init__(self, fit_intercept=True):
        self.coefficients = None
        self.intercept = None
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        Fit linear model using normal equation.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        X = np.array(X)
        y = np.array(y)

        try:
            self.coefficients = np.linalg.inv(X.T @ X) @ X.T @ y
            if self.fit_intercept:
                self.intercept = np.mean(y - X @ self.coefficients)
            else:
                self.intercept = 0
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            self.coefficients = np.linalg.pinv(X.T @ X) @ X.T @ y
            if self.fit_intercept:
                self.intercept = np.mean(y - X @ self.coefficients)
            else:
                self.intercept = 0

    def predict(self, X):
        """
        Predict using the linear model.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns:
        --------
        array, shape (n_samples,)
            Predicted values
        """
        X = np.array(X)
        return X @ self.coefficients + self.intercept

    def save_weights(self, filepath):
        """Save model weights to file."""
        weights = {
            'coefficients': self.coefficients,
            'intercept': self.intercept,
            'fit_intercept': self.fit_intercept
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(weights, f)
        print(f"Model weights saved to {filepath}")

    def load_weights(self, filepath):
        """Load model weights from file."""
        with open(filepath, 'rb') as f:
            weights = pickle.load(f)
        self.coefficients = weights['coefficients']
        self.intercept = weights['intercept']
        self.fit_intercept = weights['fit_intercept']
        print(f"Model weights loaded from {filepath}")


class LinearRegressionGD:
    """
    Linear Regression using Gradient Descent optimization.

    Parameters:
    -----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    n_iterations : int, default=1000
        Number of iterations for optimization
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.coefficients = None
        self.intercept = None
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.loss_history = []

    def fit(self, X, y):
        """
        Fit linear model using gradient descent.

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

        # Initialize parameters with zeros (random initialization for proper gradient descent)
        self.coefficients = np.zeros(n_features)
        self.intercept = 0

        for iteration in range(self.n_iterations):
            y_predicted = self.predict(X)

            # Calculate gradients
            residuals = y_predicted - y
            gradient_coefficients = (2 / n_samples) * X.T @ residuals
            gradient_intercept = (2 / n_samples) * np.sum(residuals)

            # Update parameters
            self.coefficients -= self.learning_rate * gradient_coefficients
            self.intercept -= self.learning_rate * gradient_intercept

            # Track loss (MSE)
            loss = np.mean(residuals ** 2)
            self.loss_history.append(loss)

    def predict(self, X):
        """
        Predict using the linear model.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples

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
        plt.plot(range(len(self.loss_history)), self.loss_history, label='MSE Loss')
        plt.xlabel("Iteration")
        plt.ylabel("Loss (MSE)")
        plt.title("Loss Curve: Linear Regression (Gradient Descent)")
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
        weights = {
            'coefficients': self.coefficients,
            'intercept': self.intercept,
            'learning_rate': self.learning_rate,
            'n_iterations': self.n_iterations,
            'loss_history': self.loss_history
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(weights, f)
        print(f"Model weights saved to {filepath}")

    def load_weights(self, filepath):
        """Load model weights from file."""
        with open(filepath, 'rb') as f:
            weights = pickle.load(f)
        self.coefficients = weights['coefficients']
        self.intercept = weights['intercept']
        self.learning_rate = weights['learning_rate']
        self.n_iterations = weights['n_iterations']
        self.loss_history = weights.get('loss_history', [])
        print(f"Model weights loaded from {filepath}")
