import numpy as np

class MLPFromScratch:
    def __init__(self, layer_sizes, learning_rate=0.01, num_iterations=1000):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.xp = np
        
        self.weights = []
        self.biases = []
        self.losses = []

        # Initialize weights and biases for each layer
        for i in range(len(layer_sizes) - 1):
            # Use He initialization for ReLU activation: sqrt(2/n_in)
            # For hidden layers, use He; for output layer, use Xavier
            if i < len(layer_sizes) - 2:  # Hidden layers
                scale = np.sqrt(2.0 / layer_sizes[i])
            else:  # Output layer
                scale = np.sqrt(1.0 / layer_sizes[i])

            weight_matrix = np.random.randn(layer_sizes[i], layer_sizes[i + 1]).astype(np.float32) * scale
            bias_vector = np.zeros((1, layer_sizes[i + 1]), dtype=np.float32)

            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def relu(self, z):
        """ReLU activation for hidden layers."""
        return self.xp.maximum(0, z)

    def relu_derivative(self, z):
        """Derivative of ReLU."""
        return (z > 0).astype(self.xp.float32)

    def sigmoid(self, z):
        """Sigmoid activation (for backward compatibility)."""
        z_clipped = self.xp.clip(z, -500, 500)
        return 1 / (1 + self.xp.exp(-z_clipped))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def softmax(self, z):
        """Softmax activation for multi-class output layer."""
        # Subtract max for numerical stability
        z_shifted = z - self.xp.max(z, axis=1, keepdims=True)
        exp_z = self.xp.exp(z_shifted)
        return exp_z / self.xp.sum(exp_z, axis=1, keepdims=True)

    def predict_proba(self, X):
        """Return class probabilities for each sample."""
        X = np.asarray(X, dtype=np.float32)

        a = X
        # Hidden layers use ReLU
        for i, (w, b) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
            z = self.xp.dot(a, w) + b
            a = self.relu(z)

        # Output layer - sigmoid for binary, softmax for multi-class
        z_output = self.xp.dot(a, self.weights[-1]) + self.biases[-1]
        if hasattr(self, 'is_binary') and self.is_binary:
            a_output = self.sigmoid(z_output)
        else:
            a_output = self.softmax(z_output)

        return a_output

    def predict(self, X):
        """Return predicted class labels."""
        probabilities = self.predict_proba(X)
        if hasattr(self, 'is_binary') and self.is_binary:
            return (probabilities >= 0.5).astype(int)
        else:
            return self.xp.argmax(probabilities, axis=1)

    def compute_loss(self, y_true, y_pred):
        """Cross-entropy loss for binary or multi-class classification."""
        # Clip predictions to prevent log(0)
        y_pred_clipped = self.xp.clip(y_pred, 1e-7, 1 - 1e-7)

        # Binary classification loss
        if hasattr(self, 'is_binary') and self.is_binary:
            loss = -self.xp.mean(
                y_true * self.xp.log(y_pred_clipped) +
                (1 - y_true) * self.xp.log(1 - y_pred_clipped)
            )
        else:
            # Multi-class loss
            # If y_true is one-hot encoded
            if len(y_true.shape) > 1 and y_true.shape[1] > 1:
                loss = -self.xp.mean(self.xp.sum(y_true * self.xp.log(y_pred_clipped), axis=1))
            else:
                # If y_true is class indices, convert to one-hot
                n_samples = y_true.shape[0]
                n_classes = y_pred.shape[1]
                y_true_one_hot = self.xp.zeros((n_samples, n_classes), dtype=self.xp.float32)
                y_true_one_hot[self.xp.arange(n_samples), y_true.astype(int).flatten()] = 1
                loss = -self.xp.mean(self.xp.sum(y_true_one_hot * self.xp.log(y_pred_clipped), axis=1))

        return float(loss)  # Convert to Python float for compatibility

    def fit(self, X, y):
        # Convert inputs to numpy arrays
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        n_samples = X.shape[0]
        n_classes = self.layer_sizes[-1]

        # Check if this is binary classification (1 output neuron) or multi-class
        if n_classes == 1:
            # Binary classification mode - use sigmoid output
            self.is_binary = True
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            print(f"Training binary MLP with {X.shape[0]} samples...")
        else:
            # Multi-class classification mode - use softmax output
            self.is_binary = False
            if len(y.shape) == 1 or (len(y.shape) == 2 and y.shape[1] == 1):
                # y is class indices, convert to one-hot
                y_flat = y.flatten().astype(int)
                y_one_hot = self.xp.zeros((n_samples, n_classes), dtype=self.xp.float32)
                y_one_hot[self.xp.arange(n_samples), y_flat] = 1
                y = y_one_hot
                print(f"Converted labels to one-hot encoding: {n_samples} samples, {n_classes} classes")
            print(f"Training multi-class MLP with {X.shape[0]} samples...")

        for iteration in range(self.num_iterations):
            # Forward pass
            activations = [X]
            z_values = []
            a = X

            # Hidden layers with ReLU
            for i in range(len(self.weights) - 1):
                z = self.xp.dot(a, self.weights[i]) + self.biases[i]
                z_values.append(z)
                a = self.relu(z)
                activations.append(a)

            # Output layer - sigmoid for binary, softmax for multi-class
            z_output = self.xp.dot(a, self.weights[-1]) + self.biases[-1]
            z_values.append(z_output)
            if self.is_binary:
                a_output = self.sigmoid(z_output)
            else:
                a_output = self.softmax(z_output)
            activations.append(a_output)

            loss = self.compute_loss(y, a_output)
            self.losses.append(loss)

            # Print progress
            if (iteration + 1) % 200 == 0 or iteration == 0:
                print(f"Iteration {iteration+1}/{self.num_iterations}, Loss: {loss:.6f}")

            # Backward pass (backpropagation)
            if self.is_binary:
                # For sigmoid + binary cross-entropy, gradient is (y_pred - y_true)
                delta = a_output - y
            else:
                # For softmax + cross-entropy, gradient is (y_pred - y_true)
                delta = a_output - y

            # Backpropagate through layers
            for i in reversed(range(len(self.weights))):
                a_prev = activations[i]
                batch_size = y.shape[0]

                # Compute gradients
                dw = self.xp.dot(a_prev.T, delta) / batch_size
                db = self.xp.sum(delta, axis=0, keepdims=True) / batch_size

                # Update weights and biases
                self.weights[i] -= self.learning_rate * dw
                self.biases[i] -= self.learning_rate * db

                # Propagate error to previous layer (except for input layer)
                if i > 0:
                    delta = self.xp.dot(delta, self.weights[i].T) * self.relu_derivative(z_values[i-1])

        return self.weights, self.biases, self.losses