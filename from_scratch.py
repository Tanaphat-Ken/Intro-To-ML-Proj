import numpy as np
from math import sqrt, pi, exp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

class LinearRegressionScratch:
    """
    Custom implementation of linear regression using ordinary least squares (OLS).
    """
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        """
        Fits the linear regression model to the given data.
        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the
 target values.
        """
        # Calculate coefficients using the closed-form solution
        self.coef_ = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
        self.intercept_ = np.mean(y - np.dot(X, self.coef_))

    def predict(self, X):
        """
        Predicts the target values for new data.
        Args:
            X: A numpy array of shape (n_samples, n_features) representing the new input data.
        Returns:
            A numpy array of shape (n_samples,) representing the predicted target values.
        """
        return np.dot(X, self.coef_) + self.intercept_

class UpdatedLinearRegressionScratch:
    """
    Custom implementation of linear regression using gradient descent.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.coef_ = None
        self.intercept_ = None
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        """
        Fits the linear regression model to the given data using gradient descent.
        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the target values.
        """
        # Initialize coefficients (slope) and intercept to zero or random small values
        self.coef_ = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
        self.intercept_ = np.mean(y - np.dot(X, self.coef_))

        m = len(y)  # number of training examples

        for _ in range(self.n_iterations):
            # Calculate the predictions
            y_pred = self.predict(X)

            # Compute the residuals (errors)
            error = (1/m)*(y_pred - y)**2  # Note: just difference loss = the distance of prediction to ground truth, not a practical loss, like MSE, MAE

            # Calculate the gradient for intercept (slope) and coefficients (slope)
            intercept_gradient = (2/m) * np.sum(error)
            coef_gradient = (2/m) * np.dot(X.T, error)

            # Update the parameters using the gradients
            self.intercept_ -= self.learning_rate * intercept_gradient
            self.coef_ -= self.learning_rate * coef_gradient

    def predict(self, X):
        """
        Predicts the target values for new data.
        Args:
            X: A numpy array of shape (n_samples, n_features) representing the new input data.
        Returns:
            A numpy array of shape (n_samples,) representing the predicted target values.
        """
        return np.dot(X, self.coef_) + self.intercept_
    
class PolyRegressionScratch:
    """
    Custom implementation of linear regression using ordinary least squares (OLS).
    """
    def __init__(self):
        self.a3 = 1
        self.a2 = 1
        self.a1 = 1
        self.a0 = 1

    def predict(self, x):
        return self.a3 * x**3 + self.a2 * x**2 + self.a1 * x + self.a0

    # Cost function (MSE)
    def compute_cost(self, X_HW, y_HW):
        m = len(y_HW)
        predictions = self.predict(X_HW)
        cost = (1/(m)) * np.sum((predictions - y_HW)**2)
        return cost

    # Gradient Descent
    def fit(self, X_HW, y_HW, iterations=1000, alpha=0.01):
        m = len(y_HW)
        for _ in range(iterations):
            predictions = self.predict(X_HW)

            # gradient compute
            dJ_da0 = (2/m) * np.sum(predictions - y_HW)
            dJ_da1 = (2/m) * np.sum((predictions - y_HW) * X_HW)
            dJ_da2 = (2/m) * np.sum((predictions - y_HW) * X_HW**2)
            dJ_da3 = (2/m) * np.sum((predictions - y_HW) * X_HW**3)

            # parameter update
            self.a0 -= alpha * dJ_da0
            self.a1 -= alpha * dJ_da1
            self.a2 -= alpha * dJ_da2
            self.a3 -= alpha * dJ_da3

            # cost compute if more iteration (optional)
            if _ % 100 == 0:
                print(f"Iteration {_}, Cost: {self.compute_cost(X_HW, y_HW)}")
    
class MultiVariateLinearRegressionScratch:
    """
    การใช้งาน Multi-Variate Linear Regression แบบ from scratch
    โดยใช้ Normal Equation: β = (X^T X)^(-1) X^T y
    """
    def __init__(self):
        self.coefficients = None
        self.intercept = None
        
    def fit(self, X, y):
        """
        ฝึกฝนโมเดล
        Args:
            X: array ของ features shape (n_samples, n_features)
            y: array ของ target values shape (n_samples,)
        """
        # เพิ่ม bias column (column ของ 1) เพื่อหา intercept
        n_samples, n_features = X.shape
        X_with_bias = np.column_stack([np.ones(n_samples), X])
        
        # Normal Equation: β = (X^T X)^(-1) X^T y
        try:
            beta = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
            self.intercept = beta[0]
            self.coefficients = beta[1:]
        except np.linalg.LinAlgError:
            # ใช้ pseudo-inverse หาก matrix ไม่สามารถ inverse ได้
            beta = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
            self.intercept = beta[0]
            self.coefficients = beta[1:]
    
    def predict(self, X):
        """
        ทำนายค่า
        Args:
            X: array ของ features shape (n_samples, n_features)
        Returns:
            predictions: array ของค่าทำนาย shape (n_samples,)
        """
        return X @ self.coefficients + self.intercept
    
    def score(self, X, y):
        """
        คำนวณ R-squared
        Args:
            X: array ของ features
            y: array ของ target values ที่แท้จริง
        Returns:
            r2: ค่า R-squared
        """
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
    
class PolynomialMultiVariateRegressionScratch:
    """
    การใช้งาน Polynomial Multi-Variate Regression แบบ from scratch
    สร้าง polynomial features แล้วใช้ linear regression
    """
    def __init__(self, degree=2):
        self.degree = degree
        self.coefficients = None
        self.intercept = None
        
    def create_polynomial_features(self, X):
        """
        สร้าง polynomial features
        สำหรับ degree=2 และ 3 features จะได้:
        [1, x1, x2, x3, x1^2, x1*x2, x1*x3, x2^2, x2*x3, x3^2]
        """
        n_samples, n_features = X.shape
        
        if self.degree == 1:
            return X
        
        # เริ่มด้วย features เดิม
        poly_features = [X]
        
        # เพิ่ม quadratic terms (degree 2)
        if self.degree >= 2:
            # x_i^2
            for i in range(n_features):
                poly_features.append((X[:, i] ** 2).reshape(-1, 1))
            
            # x_i * x_j (i < j)
            for i in range(n_features):
                for j in range(i+1, n_features):
                    poly_features.append((X[:, i] * X[:, j]).reshape(-1, 1))
        
        # เพิ่ม cubic terms (degree 3)
        if self.degree >= 3:
            # x_i^3
            for i in range(n_features):
                poly_features.append((X[:, i] ** 3).reshape(-1, 1))
            
            # x_i^2 * x_j
            for i in range(n_features):
                for j in range(n_features):
                    if i != j:
                        poly_features.append((X[:, i]**2 * X[:, j]).reshape(-1, 1))
            
            # x_i * x_j * x_k (i < j < k)
            for i in range(n_features):
                for j in range(i+1, n_features):
                    for k in range(j+1, n_features):
                        poly_features.append((X[:, i] * X[:, j] * X[:, k]).reshape(-1, 1))
        
        return np.column_stack(poly_features)
    
    def fit(self, X, y):
        """
        ฝึกฝนโมเดล
        """
        # สร้าง polynomial features
        X_poly = self.create_polynomial_features(X)
        
        # เพิ่ม bias column
        n_samples = X_poly.shape[0]
        X_with_bias = np.column_stack([np.ones(n_samples), X_poly])
        
        # Normal Equation
        try:
            beta = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
        
        self.intercept = beta[0]
        self.coefficients = beta[1:]
        
    def predict(self, X):
        """
        ทำนายค่า
        """
        X_poly = self.create_polynomial_features(X)
        return X_poly @ self.coefficients + self.intercept
    
    def score(self, X, y):
        """
        คำนวณ R-squared
        """
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
    
class LogisticRegressionScratchMSE:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.w = None
        self.b = None
        self.losses = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        probs = self.sigmoid(z)

        # Return the class with the highest probability
        return np.where(probs >= 0.5, 1, 0)

    def initialize_weights(self, n_features):
        self.w = np.zeros((n_features, 1))  # Init with the same column number as feature
        self.b = 0

    def cost_function(self, h, y):
        m = len(y)
        # reg_term = (0.01 / (2 * m)) * np.sum(self.w ** 2)
        cost = (1/m) * np.sum((h - y)**2)

        return cost #+ reg_term

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)  # Ensure y is a column vector
        print(X.shape, y.shape)
        m = len(y)
        n_features = X.shape[1]
        self.initialize_weights(n_features)

        for i in range(self.num_iterations):
            # Forward prop
            probs = self.predict(X)

            # Cost
            # error = -(1 / m) * np.sum(y * np.log(probs + 1e-8) + (1 - y) * np.log(1 - probs + 1e-8))
            error = self.cost_function(probs, y)
            self.losses.append(error)

            # Calculate the gradient of the error with respect to the weights
            gradient_w = (1 / m) * np.dot(X.T, (probs - y))
            gradient_b = (1 / m) * np.sum(probs - y)

            # Update the weights using the gradient and the learning rate
            self.w -= self.learning_rate * gradient_w
            self.b -= self.learning_rate * gradient_b

            # cost compute if more iteration (optional)
            if i % 100 == 0:
                print(f"Iteration {i}, Cost: {error}")

    def plot_loss(self):
        plt.figure(figsize=(8, 5))
        plt.plot(range(self.num_iterations), self.losses, label='MSE Loss')
        plt.xlabel("Iteration")
        plt.ylabel("Loss (MSE)")
        plt.title("Loss Curve: Logistic Regression (MSE)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

class LogisticRegressionScratchBCE:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.w = None
        self.b = None
        self.losses = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        probs = self.sigmoid(z)

        # Return the class with the highest probability
        return np.where(probs >= 0.5, 1, 0)

    def initialize_weights(self, n_features):
        self.w = np.zeros((n_features, 1))  # Init with the same column number as feature
        self.b = 0

    def cost_function(self, h, y):
        m = len(y)
        # reg_term = (0.01 / (2 * m)) * np.sum(self.w ** 2)
        cost = -(1 / m) * np.sum(y * np.log(h + 1e-8) + (1 - y) * np.log(1 - h + 1e-8))

        return cost #+ reg_term

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)  # Ensure y is a column vector
        print(X.shape, y.shape)
        m = len(y)
        n_features = X.shape[1]
        self.initialize_weights(n_features)

        for i in range(self.num_iterations):
            # Forward prop
            probs = self.predict(X)

            # Cost
            # error = -(1 / m) * np.sum(y * np.log(probs + 1e-8) + (1 - y) * np.log(1 - probs + 1e-8))
            error = self.cost_function(probs, y)
            self.losses.append(error)

            # Calculate the gradient of the error with respect to the weights
            gradient_w = (1 / m) * np.dot(X.T, (probs - y))
            gradient_b = (1 / m) * np.sum(probs - y)

            # Update the weights using the gradient and the learning rate
            self.w -= self.learning_rate * gradient_w
            self.b -= self.learning_rate * gradient_b

            # cost compute if more iteration (optional)
            if i % 100 == 0:
                print(f"Iteration {i}, Cost: {error}")

    def plot_loss(self):
        plt.figure(figsize=(8, 5))
        plt.plot(range(self.num_iterations), self.losses, label='MSE Loss')
        plt.xlabel("Iteration")
        plt.ylabel("Loss (MSE)")
        plt.title("Loss Curve: Logistic Regression (BCE)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

class DecisionNode:
  def __init__(self, impurity=None, feature_index=None, threshold=None, left=None, right=None):
    self.left = left
    self.right = right
    # The largest impurity value of this node
    self.impurity = impurity
    # Index of the feature which make the best fit for this node.
    self.feature_index = feature_index
    # The threshold value for that feature to make the split.
    self.threshold = threshold

class LeafNode:
  def __init__(self, value):
    self.prediction_value = value

class DecisionTreeClassifierFromScratch:
  def __init__(self, min_sample_split=3, min_impurity=1e-7, max_depth=10, criterion='gini'):
    self.root = None
    self.min_sample_split = min_sample_split
    self.min_impurity = min_impurity
    self.max_depth = max_depth
    self.impurity_function = self._claculate_information_gain
    if criterion == 'entropy':
      self.criterion = self._entropy
      self.criterion_name = criterion
    else:
      self.criterion = self._gini_index
      self.criterion_name = 'gini'

  def _gini_index(self, y):
    gini = 1
    unique_value = np.unique(y)
    for val in unique_value:
      # probability of that class.
      p = np.sum(y == val) / len(y)
      gini += -np.square(p)
    return gini

  def _entropy(self, y):
    entropy = 0
    unique_value = np.unique(y)
    for val in unique_value:
      # probability of that class.
      p = np.sum(y == val) / len(y)
      entropy += -p * np.log2(p)
    return entropy

  def _claculate_information_gain(self, y, y1, y2):
    # :param y: target value.
    # :param y1: target value for dataset in the true split/right branch.
    # :param y2: target value for dataset in the false split/left branch.

    # propobility of true values.
    p = len(y1) / len(y)
    info_gain = self.criterion(y) - p * self.criterion(y1) - (1 - p) * self.criterion(y2)
    return info_gain

  def _leaf_value_calculation(self, y):
    most_frequent_label = None
    max_count = 0
    unique_labels = np.unique(y)
    # iterate over all the unique values and find their frequentcy count.
    for label in unique_labels:
      count = len( y[y == label])
      if count > max_count:
        most_frequent_label = label
        max_count = count
    return most_frequent_label

  def _partition_dataset(self, Xy, feature_index, threshold):
    col = Xy[:, feature_index]
    X_1 = Xy[col >= threshold]
    X_2 = Xy[col < threshold]

    return X_1, X_2

  def _find_best_split(self, Xy):
    best_question = tuple()
    best_datasplit = {}
    largest_impurity = 0
    n_features = (Xy.shape[1] - 1)
    # iterate over all the features.
    for feature_index in range(n_features):
      # find the unique values in that feature.
      unique_value = set(s for s in Xy[:,feature_index])
      # iterate over all the unique values to find the impurity.
      for threshold in unique_value:
        # split the dataset based on the feature value.
        true_xy, false_xy = self._partition_dataset(Xy, feature_index, threshold)

        # skip the node which has any on type 0. because this means it is already pure.
        if len(true_xy) > 0 and len(false_xy) > 0:
          # find the y values.
          y = Xy[:, -1]
          true_y = true_xy[:, -1]
          false_y = false_xy[:, -1]
          # calculate the impurity function.
          impurity = self.impurity_function(y, true_y, false_y)

          # if the calculated impurity is larger than save this value for comaparition (highest gain).
          if impurity > largest_impurity:
            largest_impurity = impurity
            best_question = (feature_index, threshold)
            best_datasplit = {
              "leftX": true_xy[:, :n_features],   # X of left subtree
              "lefty": true_xy[:, n_features:],   # y of left subtree
              "rightX": false_xy[:, :n_features],  # X of right subtree
              "righty": false_xy[:, n_features:]   # y of right subtree
            }

    return largest_impurity, best_question, best_datasplit

  def _build_tree(self, X, y, current_depth=0):
    n_samples , n_features = X.shape
    # Add y as last column of X
    Xy = np.column_stack((X, y))
    # find the Information gain on each feature each values and return the question which splits the data very well
    if (n_samples >= self.min_sample_split) and (current_depth < self.max_depth):
      # find the best split/ which question split the data well.
      impurity, question, best_datasplit = self._find_best_split(Xy)
      if impurity > self.min_impurity:
        # Build subtrees for the right and left branch.
        true_branch = self._build_tree(best_datasplit["leftX"], best_datasplit["lefty"], current_depth + 1)
        false_branch = self._build_tree(best_datasplit["rightX"], best_datasplit["righty"], current_depth + 1)
        return DecisionNode(impurity=impurity, feature_index=question[0], threshold=question[1],
                            left=true_branch, right=false_branch)

    leaf_value = self._leaf_value_calculation(y)
    return LeafNode(value=leaf_value)

  def fit(self, X, y):
    self.root = self._build_tree(X, y, current_depth=0)

  def predict_sample(self, x, tree=None):
    if isinstance(tree , LeafNode):
      return tree.prediction_value

    if tree is None:
      tree = self.root
    feature_value = x[tree.feature_index]
    branch = tree.right

    if isinstance(feature_value, int) or isinstance(feature_value, float):
      if feature_value >= tree.threshold:
        branch = tree.left
    elif feature_value == tree.threshold:
      branch = tree.left

    return self.predict_sample(x, branch)

  def predict(self, test_X):
    x = np.array(test_X)
    y_pred = [self.predict_sample(sample) for sample in x]
    y_pred = np.array(y_pred)
    return y_pred

  def draw_tree(self):
    self._draw_tree(self.root)

  def _draw_tree(self, tree = None, indentation = " ", depth=0):
    if isinstance(tree , LeafNode):
      print(indentation,"The predicted value -->", tree.prediction_value)
      return
    else:
      print(indentation,f"({depth}) Is {tree.feature_index}>={tree.threshold}?"
            f": {self.criterion_name}:{tree.impurity:.2f}")
      if tree.left is not None:
          print (indentation + '----- True branch :)')
          self._draw_tree(tree.left, indentation + "  ", depth+1)
      if tree.right is not None:
          print (indentation + '----- False branch :)')
          self._draw_tree(tree.right, indentation + "  ", depth+1)

class RandomForestClassifierFromScratch:
  def __init__(self, max_feature=None, n_trees=100, min_sample_split=2, min_impurity=1e-7, max_depth=10, criterion='gini'):
    # Initialize the trees.
    self.trees = []
    for _ in range(n_trees):
      self.trees.append(DecisionTreeClassifierFromScratch(min_sample_split=min_sample_split,min_impurity=min_impurity,
                                                          max_depth=max_depth,criterion=criterion))

    self.tree_feature_indexes = []
    # Number of trees/estimetors.
    self.n_estimators = n_trees
    # How many features can be used for a tree from the whole features.
    self.max_features = max_feature
    # Aggication function to find the prediction.
    self.prediction_aggrigation_calculation = self._maximum_vote_calculation

  def _maximum_vote_calculation(self, y_preds):
    # Find which prediction class has higest frequency in all tree prediction for each sample.
    # create a empty array to store the prediction.
    y_pred = np.empty((y_preds.shape[0], 1))
    # iterate over all the data samples.
    for i, sample_predictions in enumerate(y_preds):
      y_pred[i] = np.bincount(sample_predictions.astype('int')).argmax()

    return y_pred

  def _make_random_subset(self, X, y, n_subsets, replacement=True):
    # Create a random subset of dataset with/without replacement.
    subset = []
    # use 100% of data when replacement is true , use 50% otherwise.
    sample_size = (X.shape[0] if replacement else (X.shape[0] // 2))

    # Add y as last column of X
    Xy = np.column_stack((X, y))
    np.random.shuffle(Xy)
    # Select randome subset of data with replacement.
    for i in range(n_subsets):
      index = np.random.choice(range(sample_size), size=np.shape(range(sample_size)), replace=replacement)
      X = Xy[index][:, :-1]
      y = Xy[index][: , -1]
      subset.append({"X" : X, "y": y})
    return subset

  def fit(self, X, y):
    # if the max_features is not given then select it as square root of no on feature availabe.
    n_features = X.shape[1]
    if self.max_features == None:
      self.max_features = int(round(np.sqrt(n_features)))

    # Split the dataset into number of subsets equal to n_estimators.
    subsets = self._make_random_subset(X, y, self.n_estimators)

    for i, subset in enumerate(subsets):
      X_subset , y_subset = subset["X"], subset["y"]
      # select a random sucset of features for each tree. This is called feature bagging.
      idx = np.random.choice(range(n_features), size=self.max_features, replace=True)
      # track this for prediction.
      self.tree_feature_indexes.append(idx)
      # Get the X with the selected features only.
      X_subset = X_subset[:, idx]

      # change the y_subet to i dimentional array.
      y_subset = np.expand_dims(y_subset, axis =1)
      # build the model with selected features and selected random subset from dataset.
      self.trees[i].fit(X_subset, y_subset)

  def predict(self, test_X):
    y_preds = np.empty((test_X.shape[0], self.n_estimators))
    # find the prediction from each tree for each samples
    for i, tree in enumerate(self.trees):
      features_index = self.tree_feature_indexes[i]
      X_selected_features = test_X[:, features_index]
      if isinstance(tree, DecisionTreeClassifierFromScratch):
        y_preds[:, i] = tree.predict(X_selected_features).reshape((-1,))
      else:
        y_preds[:, i] = tree.predict(X_selected_features)
    # find the aggregated output.
    y_pred = self.prediction_aggrigation_calculation(y_preds)

    return y_pred
  
class SVM_Linear_Scratch:
    def __init__(self, C=1, batch_size=100, learning_rate=0.001, iterations=1000):
        # C = error term
        self.C = C
        self.batch_size = batch_size
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.w = None
        self.b = None

    def decision_function(self, X):
        return np.dot(X, self.w) + self.b # w.x + b

    def hingeloss(self, w, b, x, y):
        # Regularizer term
        reg = 0.5 * (w * w)

        for i in range(x.shape[0]):
            # Optimization term
            opt_term = y[i] * ((np.dot(w, x[i])) + b)
            loss = reg + self.C * max(0, 1 - opt_term)

        return loss[0]

    def fit(self, X, Y):
        # initialize
        n_samples = X.shape[0]
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0
        losses = []

        # convert y to signed value (-1, +1)
        Y = np.where(Y <= 0, -1, 1)

        # gradient descent optimization start
        for i in range(self.iterations):
            l = self.hingeloss(self.w, self.b, X, Y)
            losses.append(l)

            # iterate through samples with batch_size as interval
            for batch_start in range(0, n_samples, self.batch_size):
                gradw = 0
                gradb = 0
                for x in range(batch_start, batch_start + self.batch_size):
                    if x >= n_samples:
                        break
                     # correct classification
                    if Y[x] * self.decision_function(X[x]) >= 1:
                        gradw += 0 # w = w - α*w
                        gradb += 0  # b = b
                    # misclassification
                    else:
                        gradw += self.C * Y[x] * X[x]  # w = w - α*(w - C*yi*xi)
                        gradb += self.C * Y[x] # b = b - α*(C*yi)

                # Updating weights and bias
                self.w += self.learning_rate * gradw
                self.b += self.learning_rate * gradb

        return self.w, self.b, losses

    def predict(self, X):
        prediction = self.decision_function(X)
        label_signs = np.sign(prediction)
        result = np.where(label_signs <= -1, 0, 1)
        return result

class SVM_Non_Linear_Scratch:
    def __init__(self, kernel='poly', C=1, degree=2, const=1, sigma=0.1, iterations=1000, learning_rate= 0.001):
        self.X = None
        self.y = None
        self.alpha = None
        self.ones = None
        self.b = 0
        self.C = C
        self.iterations = iterations
        self.learning_rate = learning_rate

        if kernel == 'poly':
            self.kernel = self.polynomial_kernel
            self.degree = degree
            self.const = const
        elif kernel == 'rbf':
            self.kernel =  self.gaussian_kernel
            self.sigma = sigma

    def polynomial_kernel(self, X, Z):
        # K(X, Z) = (c + X.Z)^degree
        return (self.const + X.dot(Z.T))**self.degree

    def gaussian_kernel(self, X, Z):
        # K(X, Z) = e^( -(1/ σ2) * ||X-Z||^2 )
        return np.exp(-(1 / self.sigma ** 2) * np.linalg.norm(X[:, np.newaxis] - Z[np.newaxis, :], axis=2) ** 2)

    def decision_function(self, X):
        # ŷ = sign( (αi*yi).K(xi, xi) + b )
        return (self.alpha * self.y).dot(self.kernel(self.X, X)) + self.b

    def fit(self, X, y):
        y = np.where(y <= 0, -1, 1)
        self.X = X
        self.y = y
        self.alpha = np.random.random(X.shape[0])
        self.ones = np.ones(X.shape[0])
        self.b = 0
        losses = []

        # (yi*yj) * K(xi, xj)
        kernel_mat = np.outer(y, y) * self.kernel(X, X)

        for i in range(self.iterations):
            # 1 – yk * ∑( αj*yj * K(xj, xk) )
            gradient = self.ones - kernel_mat.dot(self.alpha)
            # α = α + η*(1 – yk * ∑( αj*yj * K(xj, xk) )) update as per gradient descent rule
            self.alpha = self.alpha + self.learning_rate * gradient
            # 0 < α < C
            self.alpha[self.alpha > self.C] = self.C
            self.alpha[self.alpha < 0] = 0
            # ∑( αi – (1/2) * ∑i( ∑j( αi*αj * (yi*yj) * K(xi, xj) ) ) )
            loss = np.sum(self.alpha) - 0.5 * np.sum(np.outer(self.alpha, self.alpha) * kernel_mat)
            losses.append(loss)

        # for bias, only consider α which 0 < α < C
        # b = avg(0≤αi≤C){ yi – ∑( αj*yj * K(xj, xi) ) }
        index = np.where((self.alpha) > 0 & (self.alpha < self.C))[0]
        b_ind = y[index] - (self.alpha * y).dot(self.kernel(X, X[index]))
        self.b = np.mean(b_ind)

        return self.alpha, self.b, losses

    def predict(self, X):
        prediction = self.decision_function(X)
        label_signs = np.sign(prediction)
        result = np.where(label_signs <= -1, 0, 1)
        return result    
    
class SVM_Linear_Scratch_Simple:
    def __init__(self, C=1, iterations=1000, lr=0.001, lambdaa=0.01):
        self.C = C
        self.lambdaa = lambdaa
        self.iterations = iterations
        self.lr = lr
        self.w = None
        self.b = None

    def decision_function(self, X):
        return np.dot(X, self.w) - self.b

    def gradient_descent(self, X, y):
        # Updates the weights and bias using gradient descent.
        y_ = np.where(y <= 0, -1, 1)
        for i, x in enumerate(X):
            if y_[i] * self.decision_function(x) >= 1:
                dw = 2 * self.lambdaa * self.w # w = w - α * (2λw)
                db = 0 # b = b
            else:
                dw = 2 * self.lambdaa * self.w - self.C * np.dot(x, y_[i]) # w = w + α * (2λw - yixi)
                db = self.C * y_[i] # b = b - α * (yi)
            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0

        for i in range(self.iterations):
            self.gradient_descent(X, y)

        return self.w, self.b

    def predict(self, X):
        output = self.decision_function(X)
        label_signs = np.sign(output)
        #set predictions to 0 if they are less than or equal to -1 else set them to 1
        predictions = np.where(label_signs <= -1, 0, 1)
        return predictions
    
class KMeansClusteringScratch:
    """
    Basic implementation of the K-Means clustering algorithm.
    """

    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        if n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer")
        if max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        if tol < 0:
            raise ValueError("tol must be non-negative")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        self._is_fitted = False

    def fit(self, X):
        X = np.array(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("Input data must be a 2D array")

        n_samples, _ = X.shape
        if n_samples == 0:
            raise ValueError("Input data must contain at least one sample")
        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot be greater than the number of samples")

        rng = np.random.default_rng(self.random_state)
        initial_indices = rng.choice(n_samples, self.n_clusters, replace=False)
        centroids = X[initial_indices]

        for iteration in range(1, self.max_iter + 1):
            distances = self._compute_distances(X, centroids)
            labels = np.argmin(distances, axis=1)

            new_centroids = centroids.copy()
            for cluster_idx in range(self.n_clusters):
                members = X[labels == cluster_idx]
                if len(members) == 0:
                    # Reinitialize empty cluster to a random data point
                    new_centroid = X[rng.integers(0, n_samples)]
                else:
                    new_centroid = members.mean(axis=0)
                new_centroids[cluster_idx] = new_centroid

            centroid_shift = np.linalg.norm(new_centroids - centroids, axis=1).max()
            centroids = new_centroids

            if centroid_shift <= self.tol:
                self.n_iter_ = iteration
                break
        else:
            # Loop did not break naturally
            distances = self._compute_distances(X, centroids)
            labels = np.argmin(distances, axis=1)
            self.n_iter_ = self.max_iter

        self.cluster_centers_ = centroids
        self.labels_ = labels
        self.inertia_ = float(np.sum((X - centroids[labels]) ** 2))
        self._is_fitted = True
        return self

    def predict(self, X):
        self._check_is_fitted()
        X = np.array(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("Input data must be a 2D array")
        distances = self._compute_distances(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def transform(self, X):
        self._check_is_fitted()
        X = np.array(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("Input data must be a 2D array")
        return self._compute_distances(X, self.cluster_centers_)

    def _compute_distances(self, X, centroids):
        # Efficient computation of squared Euclidean distances
        diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        return np.linalg.norm(diff, axis=2)

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise ValueError("This KMeansClusteringScratch instance is not fitted yet. Call 'fit' first.")

class AgglomerativeClusteringScratch:
  def __init__(self, n_clusters=None, linkage='average'):
    # Use euclidean matric for computing pairwise distance matrix
    self.n_clusters = n_clusters

    if linkage == 'complete':
      self.linkage_distance_func = self.max_linkage_distance
    elif linkage == 'single':
      self.linkage_distance_func = self.min_linkage_distance
    elif linkage == 'average':
      self.linkage_distance_func = self.avg_linkage_distance
    elif linkage == 'ward':
      self.linkage_distance_func = self.ward_method_distance

  def avg_linkage_distance(self, cluster_A, cluster_B):
    # Compute average linkage distance between two clusters
    distance = 0
    for i in range(cluster_A.shape[0]):
      distance += np.linalg.norm(cluster_B - cluster_A[i, :], axis=1).sum()
    distance /= (cluster_A.shape[0] * cluster_B.shape[0])
    return distance

  def max_linkage_distance(self, cluster_A, cluster_B):
    # Compute maximum linkage distance between two clusters
    distance = 0
    for i in range(cluster_A.shape[0]):
      distance = np.append(np.linalg.norm(cluster_B - cluster_A[i, :], axis=1), distance).max()
    return distance

  def min_linkage_distance(self, cluster_A, cluster_B):
    # Compute minimum linkage distance between two clusters
    distance = np.inf
    for i in range(cluster_A.shape[0]):
      distance = np.append(np.linalg.norm(cluster_B - cluster_A[i, :], axis=1), distance).min()
    return distance

  def ward_method_distance(self, cluster_A, cluster_B):
    # Compute the Ward linkage distance between two clusters
    n_A = cluster_A.shape[0]
    n_B = cluster_B.shape[0]

    centroid_A = np.mean(cluster_A, axis=0)
    centroid_B = np.mean(cluster_B, axis=0)

    # Distance is proportional to the squared Euclidean distance between centroids
    # scaled by size of the clusters
    diff = centroid_A - centroid_B
    distance = (n_A * n_B) / (n_A + n_B) * np.dot(diff, diff)

    return distance

  def pairwise_distance(self, data, n_samples):
    # Compute the pairwise distance matrix in euclidean matric
    distance_mat = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
      for j in range(i + 1, n_samples):
        distance = np.linalg.norm(data[i] - data[j])
        distance_mat[i, j] = distance
        distance_mat[j, i] = distance
    return distance_mat

  def update(self, data, distance_mat, labels):
      #"Find closest clusters, merge clusters, delete cluster, update distance"
      idx_upper = np.triu_indices(distance_mat.shape[0], k=1)  # Index of upper part of distance matrix (skip diagonal)
      min_value = np.min(distance_mat[idx_upper])  # Value of idx_upper
      row, col = np.argwhere(distance_mat == min_value)[0]  # Index of min_value (same as d_kl)

      # Update label
      labels[labels == col] = row
      labels[labels > col] -= 1

      # Deleted the row and column 'col'
      distance_mat = np.delete(distance_mat, col, 0)
      distance_mat = np.delete(distance_mat, col, 1)

      # Update distance matrix
      for i in range(len(distance_mat)):
          distance_mat[row, i] = self.linkage_distance_func(data[labels == row], data[labels == i])
          distance_mat[i, row] = distance_mat[row, i]
      return distance_mat, labels

  def fit_predict(self, X):
    self.data = X
    self.n_samples = self.data.shape[0]
    self.initial_distance = self.pairwise_distance(self.data, self.n_samples)
    self.labels = np.arange(self.n_samples)
    self.distance_matrix = self.initial_distance.copy()
    while len(np.unique(self.labels)) > self.n_clusters:
      # Fill in the diagonal as infinity to determine that the distance is the same position.
      np.fill_diagonal(self.distance_matrix, np.inf)
      self.distance_matrix, self.labels = self.update(self.data, self.distance_matrix, self.labels)

    return self.labels

class NaiveBayesClassifier:
    def __init__(self, categorical_features, numerical_features, target_feature, use_add1_smoothing=True):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.target_feature = target_feature
        self.use_add1_smoothing = use_add1_smoothing

        self.classes = []
        self.prior_probs = {} # P(Class)
        self.likelihoods_categorical = {} # P(Feature_cat | Class) with smoothing
        self.numerical_features_stats = {} # {Class: {Feature_num: {'mean': .., 'std': ..}}}
        self.feature_possible_values = {} # For smoothing denominator

    def fit(self, df):
        """
        ฝึกโมเดล Naive Bayes โดยคำนวณ Prior, Likelihoods, และ Stats สำหรับ Numerical Features
        **Likelihoods สำหรับ Categorical Predictors จะไม่ใช้ Add-1 Smoothing**
        """
        self.classes = df[self.target_feature].unique()
        total_samples = len(df)

        # 1. คำนวณ Prior Probabilities
        for cls in self.classes:
            self.prior_probs[cls] = df[df[self.target_feature] == cls].shape[0] / total_samples

        # 2. คำนวณ Likelihoods สำหรับ Categorical Predictors (ไม่มี Smoothing)
        for feature in self.categorical_features:
            self.feature_possible_values[feature] = df[feature].nunique() # ไม่ได้ใช้แล้วถ้าไม่ smoothing
            for cls in self.classes:
                if cls not in self.likelihoods_categorical:
                    self.likelihoods_categorical[cls] = {}
                self.likelihoods_categorical[cls][feature] = {}

                df_class = df[df[self.target_feature] == cls]
                total_count_class = df_class.shape[0]

                for feat_value in df[feature].unique():
                    count_feat_class = df_class[feature].value_counts().get(feat_value, 0)

                    # *** จุดที่แก้ไข 2: ใช้เงื่อนไข use_add1_smoothing ***
                    if self.use_add1_smoothing:
                        prob = (count_feat_class + 1) / (total_count_class + self.feature_possible_values[feature])
                    else:
                        if total_count_class > 0:
                            prob = count_feat_class / total_count_class
                        else:
                            prob = 0.0 # ถ้าไม่มีข้อมูลในคลาสนี้เลย ให้ Likelihood เป็น 0

                    self.likelihoods_categorical[cls][feature][feat_value] = prob

        # 3. คำนวณ Mean และ Standard Deviation สำหรับ Numerical Predictors
        for feature in self.numerical_features:
            for cls in self.classes:
                if cls not in self.numerical_features_stats:
                    self.numerical_features_stats[cls] = {}

                df_class_feature = df[df[self.target_feature] == cls][feature]
                self.numerical_features_stats[cls][feature] = {
                    'mean': df_class_feature.mean(),
                    'std': df_class_feature.std(ddof=1) # ddof=1 for sample std dev
                }
                # Handle cases where std dev is 0 (e.g., all values are same in a class)
                if self.numerical_features_stats[cls][feature]['std'] == 0:
                    self.numerical_features_stats[cls][feature]['std'] = 1e-9 # Prevent division by zero, small epsilon

    def _normal_pdf(self, x, mean, std):
        """Probability Density Function for Normal Distribution."""
        if std == 0: # Should be handled by epsilon in fit, but as a safeguard
            return 1.0 if x == mean else 0.0
        exponent = exp(-((x - mean) ** 2) / (2 * (std ** 2)))
        return (1 / (sqrt(2 * pi) * std)) * exponent

    def _calculate_unnormalized_posterior_with_terms(self, query_features, target_label):
        """
        คำนวณ Numerator (Score) สำหรับคลาสที่ระบุ พร้อมส่งกลับเทอมการคูณแต่ละตัว
        Score = P(X|c) * P(c)
        """
        score = self.prior_probs[target_label]
        terms = [f"{self.prior_probs[target_label]:.4f}"] # เก็บเฉพาะตัวเลขสำหรับส่วน (0.XX * 0.YY...)

        # สำหรับแสดง P(Yes) = ... หรือ P(No) = ... ในบรรทัดแรกของ P(X|c)*P(c)
        detailed_terms_display = [f"P({target_label}) = {self.prior_probs[target_label]:.4f}"]

        for feat_name, feat_value in query_features.items():
            if feat_name in self.categorical_features:
                p_feat_given_label = self.likelihoods_categorical[target_label][feat_name].get(feat_value, 0)
                score *= p_feat_given_label

                display_value = feat_value
                terms.append(f"{p_feat_given_label:.4f}")
                detailed_terms_display.append(f"P({feat_name}={display_value}|{target_label}) = {p_feat_given_label:.4f}")
            elif feat_name in self.numerical_features:
                mean = self.numerical_features_stats[target_label][feat_name]['mean']
                std = self.numerical_features_stats[target_label][feat_name]['std']
                p_feat_given_label = self._normal_pdf(feat_value, mean, std)
                score *= p_feat_given_label
                terms.append(f"{p_feat_given_label:.4f}")
                detailed_terms_display.append(f"PDF({feat_name}={feat_value}|{target_label}) = {p_feat_given_label:.4f}")
            else:
                # This should ideally not happen if features are correctly defined
                print(f"Warning: Feature '{feat_name}' not recognized during prediction. Skipping.")
        return score, terms, detailed_terms_display

    def predict_proba(self, query_features):
        """
        ทำนาย Posterior Probabilities สำหรับแต่ละคลาส
        คืนค่าเป็น dictionary {class: probability}
        """
        scores = {}
        # We don't need detailed terms for predict_proba, only for display
        for cls in self.classes:
            score, _, _ = self._calculate_unnormalized_posterior_with_terms(query_features, cls)
            scores[cls] = score

        total_score = sum(scores.values())

        posterior_probs = {}
        if total_score > 0:
            for cls in self.classes:
                posterior_probs[cls] = scores[cls] / total_score
        else:
            for cls in self.classes:
                posterior_probs[cls] = 0.0
            print("Warning: All unnormalized scores are zero. Posterior probabilities set to 0.0.")

        return posterior_probs

    def predict(self, query_features):
        """
        ทำนายคลาสที่มีความน่าจะเป็นสูงสุด
        """
        posterior_probs = self.predict_proba(query_features)

        if posterior_probs:
            return max(posterior_probs, key=posterior_probs.get)
        else:
            return None
        
class SLPFromScratch:
    def __init__(self, input_size, learning_rate=0.01, num_iterations=1000):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = np.random.randn(input_size) * 0.01  # Small random weights
        self.bias = 0.0
        self.losses = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_output)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)

    def compute_loss(self, y_true, y_pred):
        m = len(y_true)
        loss = np.mean((y_true - y_pred) ** 2)  # Mean Squared Error
        return loss

    def fit(self, X, y):
        for i in range(self.num_iterations):
            # Forward pass
            probabilities = self.predict_proba(X)
            loss = self.compute_loss(y, probabilities)
            self.losses.append(loss)

            # Backward pass (gradient descent)
            gradient = np.dot(X.T, (probabilities - y)) / y.size
            self.weights -= self.learning_rate * gradient
            self.bias -= self.learning_rate * np.mean(probabilities - y)
        return self.weights, self.bias, self.losses
    
class MLPFromScratch:
    def __init__(self, layer_sizes, learning_rate=0.01, num_iterations=1000):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = []
        self.biases = []
        self.losses = []

        # Initialize weights and biases for each layer
        for i in range(len(layer_sizes) - 1):
            weight_matrix = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01
            bias_vector = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def predict_proba(self, X):
        a = X
        for w, b in zip(self.weights, self.biases):
            z = np.dot(a, w) + b
            a = self.sigmoid(z)
        return a

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)

    def compute_loss(self, y_true, y_pred):
        m = len(y_true)
        loss = np.mean((y_true - y_pred) ** 2)  # Mean Squared Error
        return loss

    def fit(self, X, y):
        for iteration in range(self.num_iterations):
            # Forward pass
            activations = [X]
            a = X
            for w, b in zip(self.weights, self.biases):
                z = np.dot(a, w) + b
                a = self.sigmoid(z)
                activations.append(a)

            loss = self.compute_loss(y, a)
            self.losses.append(loss)

            # Backward pass (backpropagation)
            delta = (a - y) * self.sigmoid_derivative(a)
            for i in reversed(range(len(self.weights))):
                a_prev = activations[i]
                dw = np.dot(a_prev.T, delta) / y.size
                db = np.sum(delta, axis=0, keepdims=True) / y.size

                if i != 0:
                    delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(a_prev)
                self.weights[i] -= self.learning_rate * dw
                self.biases[i] -= self.learning_rate * db
        return self.weights, self.biases, self.losses
    
class RidgeRegressionScratch:
    def __init__(self, alpha=1.0, learning_rate=0.01, num_iterations=1000):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.losses = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.num_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            loss = (1 / (2 * n_samples)) * np.sum((y - y_pred) ** 2) + (self.alpha / 2) * np.sum(self.weights ** 2)
            self.losses.append(loss)

            dw = (-1 / n_samples) * np.dot(X.T, (y - y_pred)) + self.alpha * self.weights
            db = (-1 / n_samples) * np.sum(y - y_pred)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self.weights, self.bias, self.losses

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
class LassoRegressionScratch:
    def __init__(self, alpha=1.0, learning_rate=0.01, num_iterations=1000):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.losses = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.num_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            loss = (1 / (2 * n_samples)) * np.sum((y - y_pred) ** 2) + self.alpha * np.sum(np.abs(self.weights))
            self.losses.append(loss)

            dw = (-1 / n_samples) * np.dot(X.T, (y - y_pred)) + self.alpha * np.sign(self.weights)
            db = (-1 / n_samples) * np.sum(y - y_pred)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self.weights, self.bias, self.losses

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

class DecisionTreeRegressorFromScratch:
    # A simple Decision Tree Regressor implementation
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    class Node:
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
            self.feature_index = feature_index  # Index of the feature to split on
            self.threshold = threshold          # Threshold value to split on
            self.left = left                    # Left child node
            self.right = right                  # Right child node
            self.value = value                  # Value for leaf nodes

    def _mean_squared_error(self, y):
        if len(y) == 0:
            return 0
        mean_y = np.mean(y)
        return np.mean((y - mean_y) ** 2)

    def _best_split(self, X, y):
        best_mse = float('inf')
        best_feature_index = None
        best_threshold = None

        n_samples, n_features = X.shape

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                right_indices = X[:, feature_index] > threshold

                if len(y[left_indices]) < self.min_samples_split or len(y[right_indices]) < self.min_samples_split:
                    continue

                mse_left = self._mean_squared_error(y[left_indices])
                mse_right = self._mean_squared_error(y[right_indices])
                mse_total = (len(y[left_indices]) * mse_left + len(y[right_indices]) * mse_right) / n_samples

                if mse_total < best_mse:
                    best_mse = mse_total
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape

        if n_samples < self.min_samples_split or depth >= self.max_depth or len(np.unique(y)) == 1:
            leaf_value = np.mean(y)
            return self.Node(value=leaf_value)

        feature_index, threshold = self._best_split(X, y)

        if feature_index is None:
            leaf_value = np.mean(y)
            return self.Node(value=leaf_value)

        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold

        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return self.Node(feature_index=feature_index, threshold=threshold, left=left_child, right=right_child)

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        def _predict_sample(sample, node):
            if node.value is not None:
                return node.value
            if sample[node.feature_index] <= node.threshold:
                return _predict_sample(sample, node.left)
            else:
                return _predict_sample(sample, node.right)

        return np.array([_predict_sample(sample, self.tree) for sample in X])

class GradientBoostingScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):
        # Initialize model with mean of target values
        self.initial_prediction = np.mean(y)
        y_pred = np.full(y.shape, self.initial_prediction)

        for _ in range(self.n_estimators):
            # Compute residuals
            residuals = y - y_pred

            # Fit a decision tree to the residuals
            tree = DecisionTreeRegressorFromScratch(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, residuals)
            self.trees.append(tree)

            # Update predictions
            y_pred += self.learning_rate * tree.predict(X)

    def predict(self, X):
        y_pred = np.full((X.shape[0],), self.initial_prediction)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred
    
    def predict_with_aggregation(self, X):
        # Predict using all trees and aggregate their predictions
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        y_pred = self.initial_prediction + self.learning_rate * np.mean(all_preds, axis=0)
        return y_pred