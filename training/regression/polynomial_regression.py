import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

class PolynomialRegression:
    """
    Polynomial Regression using gradient descent optimization.

    Parameters:
    -----------
    degree : int, default=3
        Degree of the polynomial
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    n_iterations : int, default=1000
        Number of iterations
    """

    def __init__(self, degree=3, learning_rate=0.01, n_iterations=1000):
        self.degree = degree
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.coefficients = None
        self.loss_history = []

    def _polynomial_transform(self, x):
        """Transform input to polynomial features."""
        result = self.coefficients[self.degree]
        for i in range(self.degree - 1, -1, -1):
            result = result * x + self.coefficients[i]
        return result

    def predict(self, X):
        """
        Predict using polynomial model.

        Parameters:
        -----------
        X : array-like, shape (n_samples,)
            Input samples

        Returns:
        --------
        array, shape (n_samples,)
            Predicted values
        """
        X = np.array(X)
        return self._polynomial_transform(X)

    def compute_loss(self, X, y):
        """Compute Mean Squared Error loss."""
        predictions = self.predict(X)
        return np.mean((predictions - y) ** 2)

    def fit(self, X, y):
        """
        Fit polynomial regression model.

        Parameters:
        -----------
        X : array-like, shape (n_samples,)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        X = np.array(X)
        y = np.array(y)
        n_samples = len(y)

        # Initialize coefficients
        self.coefficients = np.ones(self.degree + 1)

        for iteration in range(self.n_iterations):
            predictions = self.predict(X)
            errors = predictions - y

            # Compute gradients for each coefficient
            gradients = np.zeros(self.degree + 1)
            for degree_idx in range(self.degree + 1):
                gradients[degree_idx] = (2 / n_samples) * np.sum(errors * (X ** degree_idx))

            # Update coefficients
            self.coefficients -= self.learning_rate * gradients

            # Track loss
            if iteration % 100 == 0:
                loss = self.compute_loss(X, y)
                self.loss_history.append(loss)
                print(f"Iteration {iteration}, Loss: {loss:.6f}")

    def plot_loss(self):
        """Plot the loss curve during training."""
        plt.figure(figsize=(8, 5))
        plt.plot(range(0, self.n_iterations, 100), self.loss_history, label='MSE Loss', marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("Loss (MSE)")
        plt.title("Loss Curve: Polynomial Regression")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


class MultiVariatePolynomialRegression:
    """
    Polynomial Regression for multiple features using Normal Equation.

    Parameters:
    -----------
    degree : int, default=2
        Degree of polynomial features
    """

    def __init__(self, degree=2):
        self.degree = degree
        self.coefficients = None
        self.intercept = None

    def _create_polynomial_features(self, X):
        """
        Create polynomial features up to specified degree.

        For degree=2 with features [x1, x2, x3]:
        Returns [x1, x2, x3, x1^2, x1*x2, x1*x3, x2^2, x2*x3, x3^2]
        """
        X = np.array(X)
        n_samples, n_features = X.shape

        if self.degree == 1:
            return X

        poly_features = [X]

        # Degree 2: squared terms and interaction terms
        if self.degree >= 2:
            # Squared terms: x_i^2
            for feature_idx in range(n_features):
                poly_features.append((X[:, feature_idx] ** 2).reshape(-1, 1))

            # Interaction terms: x_i * x_j where i < j
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    poly_features.append((X[:, i] * X[:, j]).reshape(-1, 1))

        # Degree 3: cubic terms and higher-order interactions
        if self.degree >= 3:
            # Cubic terms: x_i^3
            for feature_idx in range(n_features):
                poly_features.append((X[:, feature_idx] ** 3).reshape(-1, 1))

            # Squared-linear interactions: x_i^2 * x_j
            for i in range(n_features):
                for j in range(n_features):
                    if i != j:
                        poly_features.append((X[:, i] ** 2 * X[:, j]).reshape(-1, 1))

            # Three-way interactions: x_i * x_j * x_k where i < j < k
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    for k in range(j + 1, n_features):
                        poly_features.append((X[:, i] * X[:, j] * X[:, k]).reshape(-1, 1))

        return np.column_stack(poly_features)

    def fit(self, X, y):
        """
        Fit polynomial regression model using Normal Equation.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        X_poly = self._create_polynomial_features(X)
        n_samples = X_poly.shape[0]
        X_with_intercept = np.column_stack([np.ones(n_samples), X_poly])

        try:
            beta = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y

        self.intercept = beta[0]
        self.coefficients = beta[1:]

    def predict(self, X):
        """
        Predict using polynomial regression model.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input samples

        Returns:
        --------
        array, shape (n_samples,)
            Predicted values
        """
        X_poly = self._create_polynomial_features(X)
        return X_poly @ self.coefficients + self.intercept

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
