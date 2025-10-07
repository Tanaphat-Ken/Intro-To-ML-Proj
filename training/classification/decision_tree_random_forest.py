import numpy as np

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
    # OPTIMIZED: Vectorized gini calculation
    if len(y) == 0:
      return 0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    gini = 1 - np.sum(probabilities ** 2)
    return gini

  def _entropy(self, y):
    # OPTIMIZED: Vectorized entropy calculation
    if len(y) == 0:
      return 0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    # Avoid log(0)
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))
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
    # OPTIMIZED: Use bincount for faster mode calculation
    if len(y) == 0:
      return 0
    y_int = y.astype(int).flatten()
    counts = np.bincount(y_int)
    return np.argmax(counts)

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
    
    # OPTIMIZATION: Sample max 20 thresholds per feature instead of all unique values
    MAX_THRESHOLDS = 20
    
    # iterate over all the features.
    for feature_index in range(n_features):
      # find the unique values in that feature.
      unique_value = np.unique(Xy[:, feature_index])
      
      # OPTIMIZATION: If too many unique values, sample subset
      if len(unique_value) > MAX_THRESHOLDS:
        # Use percentiles for better coverage
        percentiles = np.linspace(0, 100, MAX_THRESHOLDS)
        threshold_candidates = np.percentile(Xy[:, feature_index], percentiles)
        threshold_candidates = np.unique(threshold_candidates)
      else:
        threshold_candidates = unique_value
      
      # iterate over the threshold candidates
      for threshold in threshold_candidates:
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