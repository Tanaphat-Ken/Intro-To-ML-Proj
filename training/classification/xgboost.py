"""XGBoost implementation for gradient boosting."""

import numpy as np
import pandas as pd
import math
from collections import defaultdict

# Optional progress bar (tqdm) — fallback to prints if not installed
try:
    from tqdm import trange
    TQDM_AVAILABLE = True
except Exception:
    trange = range
    TQDM_AVAILABLE = False


class BinaryCrossEntropyLoss:
    """Binary cross-entropy loss function with sigmoid activation."""

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def loss(labels, predictions):
        """Compute binary cross-entropy loss."""
        probs = BinaryCrossEntropyLoss.sigmoid(predictions)

        # To avoid log(0)
        epsilon = 1e-15
        probs = np.clip(probs, epsilon, 1 - epsilon)

        # Binary log loss
        return -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))

    @staticmethod
    def gradients(labels, predictions):
        """Compute gradients of binary cross-entropy loss."""
        probs = BinaryCrossEntropyLoss.sigmoid(predictions)

        # Gradient of binary cross-entropy
        return probs - labels

    @staticmethod
    def hessians(labels, predictions):
        """Compute hessians of binary cross-entropy loss."""
        probs = BinaryCrossEntropyLoss.sigmoid(predictions)

        # Hessian for sigmoid cross-entropy
        return probs * (1 - probs)


class BoostedTree:
    """
    A single decision tree within a gradient boosting ensemble.

    This class implements a regression tree that optimizes splits based on
    gradients and hessians from the loss function. It follows the XGBoost algorithm
    principles for building trees in a boosted ensemble.

    Parameters
    ----------
    X : numpy.ndarray or pandas.DataFrame
        The feature matrix for training
    gradients : numpy.ndarray or pandas.Series
        First-order gradients of the loss function
    hessians : numpy.ndarray or pandas.Series
        Second-order gradients (hessians) of the loss function
    params : dict
        Dictionary of hyperparameters:
        - 'min_child_weight': Minimum sum of hessian needed in a child node
        - 'reg_lambda': L2 regularization term
        - 'gamma': Minimum loss reduction to make a split
    max_depth : int
        Maximum depth of the tree
    idxs : numpy.ndarray, optional
        Indices of the samples to use for building this tree
    """
    def __init__(self, X, gradients, hessians, params, max_depth, idxs=None):
        self.X = X.values if isinstance(X, pd.DataFrame) else X
        self.gradients = gradients.values if isinstance(gradients, pd.Series) else gradients
        self.hessians = hessians.values if isinstance(hessians, pd.Series) else hessians
        self.params = params
        self.min_child_weight = self.params['min_child_weight'] if self.params['min_child_weight'] else 1.0
        self._lambda = self.params['reg_lambda'] if self.params['reg_lambda'] else 1.0
        self.gamma = self.params['gamma'] if self.params['gamma'] else 0.0
        self.max_depth = max_depth
        self.ridxs = idxs if idxs is not None else np.arange(len(gradients))
        self.num_examples = len(self.ridxs)  # Number of training examples
        self.num_features = X.shape[1]  # Number of features
        self.weight = -self.gradients[self.ridxs].sum() / (self.hessians[self.ridxs].sum() + self._lambda)  # Leaf weight
        self.split_score = 0.0  # Best gain so far
        self.split_idx = 0  # Feature index for split
        self.threshold = 0.0  # Threshold value for split
        self._build_tree_structure()  # Recursively build the tree

    def _build_tree_structure(self):
        """
        Recursively builds the tree structure by finding the best splits.

        This method attempts to find the best split for the current node by evaluating
        all possible features. If a valid split is found, it creates left and right
        child nodes recursively until max_depth is reached or no valid split is found.

        Returns
        -------
        None
        """
        if self.max_depth <= 0:
            return  # Reached max depth, stop recursion

        for fidx in range(self.num_features):
            self._find_best_split_score(fidx)  # Try splitting on each feature

        if self._is_leaf:
            return  # No valid split found, stop here

        feature = self.X[self.ridxs, self.split_idx]
        left_idxs = np.nonzero(feature <= self.threshold)[0]
        right_idxs = np.nonzero(feature > self.threshold)[0]

        # Recursively build left and right subtrees
        self.left = BoostedTree(self.X, self.gradients, self.hessians, self.params,
                                self.max_depth - 1, self.ridxs[left_idxs])
        self.right = BoostedTree(self.X, self.gradients, self.hessians, self.params,
                                 self.max_depth - 1, self.ridxs[right_idxs])

    def _find_best_split_score(self, fidx):
        """
        Finds the best splitting point for a given feature.

        This method evaluates all possible split points for the given feature and
        computes the gain for each. It updates the tree's split information if a
        better split than the current best is found.

        Parameters
        ----------
        fidx : int
            Index of the feature to evaluate for splitting

        Returns
        -------
        None
        """
        feature = self.X[self.ridxs, fidx]
        gradients = self.gradients[self.ridxs]
        hessians = self.hessians[self.ridxs]

        sorted_idxs = np.argsort(feature)
        sorted_feature = feature[sorted_idxs]
        sorted_gradient = gradients[sorted_idxs]
        sorted_hessians = hessians[sorted_idxs]

        hessian_sum = sorted_hessians.sum()
        gradient_sum = sorted_gradient.sum()

        right_hessian_sum = hessian_sum
        right_gradient_sum = gradient_sum
        left_hessian_sum = 0.0
        left_gradient_sum = 0.0

        for idx in range(0, self.num_examples - 1):
            candidate = sorted_feature[idx]
            neighbor = sorted_feature[idx + 1]

            gradient = sorted_gradient[idx]
            hessian = sorted_hessians[idx]

            right_gradient_sum -= gradient
            right_hessian_sum -= hessian
            left_gradient_sum += gradient
            left_hessian_sum += hessian

            if right_hessian_sum <= self.min_child_weight:
                return  # Stop if the right child is too small

            # Compute gain from potential split
            right_score = (right_gradient_sum ** 2) / (right_hessian_sum + self._lambda)
            left_score = (left_gradient_sum ** 2) / (left_hessian_sum + self._lambda)
            score_before_split = (gradient_sum ** 2) / (hessian_sum + self._lambda)
            gain = 0.5 * (left_score + right_score - score_before_split) - self.gamma

            # Save split if it's the best so far
            if gain > self.split_score:
                self.split_score = gain
                self.split_idx = fidx
                self.threshold = (candidate + neighbor) / 2

    def predict(self, X):
        """
        Make predictions for a batch of examples.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            The feature matrix to make predictions on

        Returns
        -------
        numpy.ndarray
            Predictions for each example
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.array([self._predict_row(example) for example in X])  # Predict each row

    def _predict_row(self, example):
        """
        Make a prediction for a single example by traversing the tree.

        This method traverses the tree from root to leaf based on the feature
        values of the given example, and returns the weight of the leaf node.

        Parameters
        ----------
        example : numpy.ndarray
            A single example as an array of feature values

        Returns
        -------
        float
            The prediction value for this example
        """
        if self._is_leaf:
            return self.weight  # Return leaf weight
        child = self.left if example[self.split_idx] <= self.threshold else self.right
        return child._predict_row(example)  # Recurse down the tree

    @property
    def _is_leaf(self):
        """
        Determines if the current node is a leaf node.

        Returns
        -------
        bool
            True if this is a leaf node (no valid split found), False otherwise
        """
        return self.split_score == 0.0  # Leaf node if no gain found


class XGBoost:
    """
    XGBoost implementation for gradient boosting.

    This class implements a gradient boosting algorithm inspired by XGBoost.
    It builds an ensemble of BoostedTree models to make predictions by
    sequentially adding trees that correct the errors of previous ones.

    Parameters
    ----------
    params : dict
        Dictionary of hyperparameters:
        - 'subsample': Fraction of training examples to use for each tree
        - 'base_score': Initial prediction value for all instances
        - 'learning_rate': Step size shrinkage used to prevent overfitting
        - 'max_depth': Maximum depth of each tree
        - Other parameters passed to BoostedTree (min_child_weight, reg_lambda, gamma)
    objective : object
        Objective function that implements gradients() and hessians() methods
    seed : int, default=42
        Random seed for reproducibility
    """
    def __init__(self, params, objective, seed=42):
        self.trees = []  # Store all trained trees
        self.params = defaultdict(lambda: None, params)  # Default values for missing params
        self.objective = objective  # Loss function
        self.subsample = self.params['subsample'] if self.params['subsample'] else 1.0
        self.base_score = self.params['base_score'] if self.params['base_score'] else 0.5
        self.learning_rate = self.params['learning_rate'] if self.params['learning_rate'] else 1e-1
        self.max_depth = self.params['max_depth'] if self.params['max_depth'] else 5
        self.rng = np.random.default_rng(seed=seed)  # Random number generator

    def fit(self, X, y, num_rounds):
        """
        Fit the XGBoost model to training data.

        Trains an ensemble of trees sequentially, where each tree attempts to
        correct the errors made by the previous trees. The process involves:
        1. Computing gradients and hessians based on current predictions
        2. Building a new tree to minimize these gradients
        3. Adding the tree's predictions (scaled by learning rate) to the ensemble

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Training features
        y : numpy.ndarray or pandas.Series
            Target values
        num_rounds : int
            Number of boosting rounds (trees) to build

        Returns
        -------
        None
        """
        predictions = self.base_score * np.ones(shape=y.shape)  # Initialize predictions

        iterator = trange(num_rounds, desc='Boosting rounds') if TQDM_AVAILABLE else range(num_rounds)
        for rnd in iterator:
            gradients = self.objective.gradients(y, predictions)  # Compute gradients
            hessians = self.objective.hessians(y, predictions)  # Compute hessians
            # Row sampling
            idxs = None if self.subsample == 1.0 else self.rng.choice(
                len(y),
                size=math.floor(self.subsample * len(y)),
                replace=False
            )
            # Train one tree on the current gradients
            tree = BoostedTree(
                X=X,
                gradients=gradients,
                hessians=hessians,
                params=self.params,
                max_depth=self.max_depth,
                idxs=idxs
            )
            self.trees.append(tree)
            predictions += self.learning_rate * tree.predict(X)  # Update predictions

    def predict(self, X):
        """
        Make predictions using the trained XGBoost model.

        Computes predictions by starting with the base score and adding
        the weighted contributions from all trees in the ensemble.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Features to make predictions on

        Returns
        -------
        numpy.ndarray
            Predicted values for each input example
        """
        # Add predictions from all trees
        return self.base_score + self.learning_rate * np.sum([tree.predict(X) for tree in self.trees], axis=0)


class XGBoostClassifier:
    """
    XGBoost classifier for binary classification problems using sigmoid activation.

    This class provides a wrapper around the XGBoost class specifically for
    binary classification. It uses a sigmoid function to convert raw model outputs
    into probability scores, which can then be thresholded to obtain binary predictions.

    Parameters
    ----------
    params : dict
        Dictionary of hyperparameters to be passed to the underlying XGBoost model.
        See XGBoost class documentation for details on supported parameters.
    threshold : float, default=0.5
        Decision threshold for binary classification. Probability scores above
        this threshold are classified as 1, otherwise 0.
    seed : int, default=42
        Random seed for reproducibility.
    """
    def __init__(self, params=None, threshold=0.5, seed=42):
        if params is None:
            params = {}
        self.params = params
        self.threshold = threshold  # Threshold to classify sigmoid output as 0 or 1
        self.objective = BinaryCrossEntropyLoss()
        self.base = XGBoost(self.params, self.objective, seed)

    def fit(self, X, y, subsample_cols=0.8, min_child_weight=1, depth=5, min_leaf=5,
            learning_rate=0.4, boosting_rounds=5, lambda_=1.5, gamma=1, eps=0.1):
        """
        Train the XGBoostClassifier.

        Trains the underlying XGBoost model using binary cross-entropy loss.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Training features.
        y : numpy.ndarray or pandas.Series
            Binary target values (0 or 1).
        subsample_cols : float, default=0.8
            Fraction of features to use for each tree
        min_child_weight : float, default=1
            Minimum sum of hessian needed in a child
        depth : int, default=5
            Maximum depth of trees
        min_leaf : int, default=5
            Minimum samples in a leaf
        learning_rate : float, default=0.4
            Learning rate for boosting
        boosting_rounds : int, default=5
            Number of boosting rounds
        lambda_ : float, default=1.5
            L2 regularization
        gamma : float, default=1
            Minimum loss reduction for split
        eps : float, default=0.1
            Approximation parameter

        Returns
        -------
        None
        """
        # Update params
        self.params.update({
            'subsample': 1.0,  # Use all samples
            'learning_rate': learning_rate,
            'max_depth': depth,
            'min_child_weight': min_child_weight,
            'reg_lambda': lambda_,
            'gamma': gamma
        })

        # Recreate base model with updated params
        self.base = XGBoost(self.params, self.objective, seed=42)
        self.base.fit(X, y, boosting_rounds)  # Train the underlying boosted trees

    def predict_proba(self, X):
        """
        Predict probability scores.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Features to make predictions on.

        Returns
        -------
        numpy.ndarray
            Probability scores for class 1.
        """
        logits = self.base.predict(X)  # Get raw scores
        probs = self.objective.sigmoid(logits)  # Apply sigmoid to get probabilities
        return probs

    def predict(self, X, threshold=None):
        """
        Make predictions using the trained XGBoostClassifier model.

        Computes raw scores using the underlying XGBoost model, then applies
        a sigmoid function to obtain probability scores and thresholds to get
        binary class labels.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Features to make predictions on.
        threshold : float, optional
            Decision threshold for binary classification. If not provided,
            uses the threshold set during initialization.

        Returns
        -------
        numpy.ndarray
            Binary class labels (0 or 1).
        """
        if threshold is None:
            threshold = self.threshold

        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
