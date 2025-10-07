import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

class MultipleLinearRegression:
    """
    Multiple Linear Regression using Normal Equation: β = (X^T X)^(-1) X^T y

    This model extends simple linear regression to handle multiple independent variables.

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
        Fit multiple linear regression model using Normal Equation.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data with multiple features
        y : array-like, shape (n_samples,)
            Target values
        """
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        if self.fit_intercept:
            # Add bias column for intercept calculation
            X_with_intercept = np.column_stack([np.ones(n_samples), X])

            try:
                # Normal Equation: β = (X^T X)^(-1) X^T y
                beta = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
                self.intercept = beta[0]
                self.coefficients = beta[1:]
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if matrix is singular
                beta = np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
                self.intercept = beta[0]
                self.coefficients = beta[1:]
        else:
            try:
                self.coefficients = np.linalg.inv(X.T @ X) @ X.T @ y
                self.intercept = 0
            except np.linalg.LinAlgError:
                self.coefficients = np.linalg.pinv(X.T @ X) @ X.T @ y
                self.intercept = 0

    def predict(self, X):
        """
        Predict using the multiple linear regression model.

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

    def score(self, X, y):
        """
        Calculate R-squared score.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        y : array-like, shape (n_samples,)
            True values

        Returns:
        --------
        float
            R-squared score
        """
        y_predicted = self.predict(X)
        return r2_score(y, y_predicted)

    def get_coefficients(self, feature_names=None):
        """
        Get model coefficients with optional feature names.

        Parameters:
        -----------
        feature_names : list, optional
            Names of features

        Returns:
        --------
        dict
            Dictionary mapping feature names to their coefficients
        """
        if feature_names is not None:
            return {
                'intercept': self.intercept,
                **{name: coef for name, coef in zip(feature_names, self.coefficients)}
            }
        else:
            return {
                'intercept': self.intercept,
                **{f'feature_{i}': coef for i, coef in enumerate(self.coefficients)}
            }


class MultipleLinearRegressionGD:
    """
    Multiple Linear Regression using Gradient Descent optimization.

    Parameters:
    -----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent. Benchmark default: 0.01
    n_iterations : int, default=1000
        Maximum training iterations. Benchmark default: 1000
    tolerance : float, default=1e-4
        Convergence threshold. Benchmark default: 1e-4
    fit_intercept : bool, default=True
        Whether to calculate intercept for the model
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000, tolerance=1e-4, fit_intercept=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.fit_intercept = fit_intercept
        self.coefficients = None
        self.intercept = None
        self.loss_history = []

    def fit(self, X, y):
        """
        Fit multiple linear regression model using gradient descent.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data with multiple features
        y : array-like, shape (n_samples,)
            Target values
        """
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        # Initialize parameters
        self.coefficients = np.zeros(n_features)
        self.intercept = 0.0 if self.fit_intercept else 0

        for iteration in range(self.n_iterations):
            # Forward pass
            y_predicted = self.predict(X)

            # Calculate gradients
            residuals = y_predicted - y
            gradient_coefficients = (2 / n_samples) * X.T @ residuals
            gradient_intercept = (2 / n_samples) * np.sum(residuals) if self.fit_intercept else 0

            # Update parameters
            self.coefficients -= self.learning_rate * gradient_coefficients
            if self.fit_intercept:
                self.intercept -= self.learning_rate * gradient_intercept

            # Track loss (MSE)
            loss = np.mean(residuals ** 2)
            self.loss_history.append(loss)

            # Early stopping with relative tolerance
            if iteration > 0:
                prev_loss = self.loss_history[-2]
                if prev_loss > 0:
                    relative_change = abs(prev_loss - loss) / prev_loss
                    if relative_change < self.tolerance:
                        print(f"Converged at iteration {iteration+1} (relative change: {relative_change:.6f})")
                        break

    def predict(self, X):
        """
        Predict using the multiple linear regression model.

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

    def score(self, X, y):
        """
        Calculate R-squared score.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        y : array-like, shape (n_samples,)
            True values

        Returns:
        --------
        float
            R-squared score
        """
        y_predicted = self.predict(X)
        return r2_score(y, y_predicted)

    def get_coefficients(self, feature_names=None):
        """
        Get model coefficients with optional feature names.

        Parameters:
        -----------
        feature_names : list, optional
            Names of features

        Returns:
        --------
        dict
            Dictionary mapping feature names to their coefficients
        """
        if feature_names is not None:
            return {
                'intercept': self.intercept,
                **{name: coef for name, coef in zip(feature_names, self.coefficients)}
            }
        else:
            return {
                'intercept': self.intercept,
                **{f'feature_{i}': coef for i, coef in enumerate(self.coefficients)}
            }

    def plot_loss(self, save_path=None):
        """Plot the loss curve during training."""
        plt.figure(figsize=(8, 5))
        plt.plot(range(len(self.loss_history)), self.loss_history, label='MSE Loss')
        plt.xlabel("Iteration")
        plt.ylabel("Loss (MSE)")
        plt.title("Loss Curve: Multiple Linear Regression (Gradient Descent)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.show()
