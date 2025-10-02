import numpy as np
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
