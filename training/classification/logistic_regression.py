import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionMSE:
    """
    Logistic Regression using Mean Squared Error loss.

    Parameters:
    -----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent. Benchmark default: 0.01
    n_iterations : int, default=1000
        Maximum training iterations. Benchmark default: 1000
    tolerance : float, default=1e-4
        Convergence threshold. Benchmark default: 1e-4
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000, tolerance=1e-4):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.coefficients = None
        self.intercept = None
        self.loss_history = []

    def _sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _initialize_parameters(self, n_features):
        """Initialize weights and bias."""
        self.coefficients = np.zeros((n_features, 1))
        self.intercept = 0

    def _compute_loss(self, probabilities, y):
        """Compute Mean Squared Error loss."""
        n_samples = len(y)
        return (1 / n_samples) * np.sum((probabilities - y) ** 2)

    def fit(self, X, y):
        """
        Fit logistic regression model using gradient descent.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values (0 or 1)
        """
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        n_samples, n_features = X.shape

        self._initialize_parameters(n_features)

        for iteration in range(self.n_iterations):
            # Forward propagation
            linear_output = X @ self.coefficients + self.intercept
            probabilities = self._sigmoid(linear_output)

            # Compute loss
            loss = self._compute_loss(probabilities, y)
            self.loss_history.append(loss)

            # Backward propagation (gradient calculation)
            errors = probabilities - y
            gradient_coefficients = (1 / n_samples) * X.T @ errors
            gradient_intercept = (1 / n_samples) * np.sum(errors)

            # Update parameters
            self.coefficients -= self.learning_rate * gradient_coefficients
            self.intercept -= self.learning_rate * gradient_intercept

            # Print progress
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss:.6f}")

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input samples

        Returns:
        --------
        array, shape (n_samples,)
            Predicted probabilities
        """
        X = np.array(X)
        linear_output = X @ self.coefficients + self.intercept
        return self._sigmoid(linear_output).flatten()

    def predict(self, X):
        """
        Predict class labels.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input samples

        Returns:
        --------
        array, shape (n_samples,)
            Predicted class labels (0 or 1)
        """
        probabilities = self.predict_proba(X)
        return np.where(probabilities >= 0.5, 1, 0)

    def plot_loss(self):
        """Plot the loss curve during training."""
        plt.figure(figsize=(8, 5))
        plt.plot(range(self.n_iterations), self.loss_history, label='MSE Loss')
        plt.xlabel("Iteration")
        plt.ylabel("Loss (MSE)")
        plt.title("Loss Curve: Logistic Regression (MSE)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


class LogisticRegressionBCE:
    """
    Logistic Regression using Binary Cross-Entropy loss.

    Parameters:
    -----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent. Benchmark default: 0.01
    n_iterations : int, default=1000
        Maximum training iterations. Benchmark default: 1000
    tolerance : float, default=1e-4
        Convergence threshold. Benchmark default: 1e-4
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000, tolerance=1e-4):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.coefficients = None
        self.intercept = None
        self.loss_history = []

    def _sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _initialize_parameters(self, n_features):
        """Initialize weights and bias."""
        self.coefficients = np.zeros((n_features, 1))
        self.intercept = 0

    def _compute_loss(self, probabilities, y):
        """Compute Binary Cross-Entropy loss."""
        n_samples = len(y)
        epsilon = 1e-8  # Small constant to prevent log(0)
        return -(1 / n_samples) * np.sum(
            y * np.log(probabilities + epsilon) + (1 - y) * np.log(1 - probabilities + epsilon)
        )

    def fit(self, X, y):
        """
        Fit logistic regression model using gradient descent.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values (0 or 1)
        """
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        n_samples, n_features = X.shape

        self._initialize_parameters(n_features)

        for iteration in range(self.n_iterations):
            # Forward propagation
            linear_output = X @ self.coefficients + self.intercept
            probabilities = self._sigmoid(linear_output)

            # Compute loss
            loss = self._compute_loss(probabilities, y)
            self.loss_history.append(loss)

            # Backward propagation (gradient calculation)
            errors = probabilities - y
            gradient_coefficients = (1 / n_samples) * X.T @ errors
            gradient_intercept = (1 / n_samples) * np.sum(errors)

            # Update parameters
            self.coefficients -= self.learning_rate * gradient_coefficients
            self.intercept -= self.learning_rate * gradient_intercept

            # Print progress
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss:.6f}")

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input samples

        Returns:
        --------
        array, shape (n_samples,)
            Predicted probabilities
        """
        X = np.array(X)
        linear_output = X @ self.coefficients + self.intercept
        return self._sigmoid(linear_output).flatten()

    def predict(self, X):
        """
        Predict class labels.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input samples

        Returns:
        --------
        array, shape (n_samples,)
            Predicted class labels (0 or 1)
        """
        probabilities = self.predict_proba(X)
        return np.where(probabilities >= 0.5, 1, 0)

    def plot_loss(self):
        """Plot the loss curve during training."""
        plt.figure(figsize=(8, 5))
        plt.plot(range(self.n_iterations), self.loss_history, label='BCE Loss')
        plt.xlabel("Iteration")
        plt.ylabel("Loss (Binary Cross-Entropy)")
        plt.title("Loss Curve: Logistic Regression (BCE)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
