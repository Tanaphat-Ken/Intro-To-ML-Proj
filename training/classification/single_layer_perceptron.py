import numpy as np

class SLPFromScratch:
    def __init__(self, input_size, learning_rate=0.01, num_iterations=1000):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.xp = np

        # Initialize weights and bias
        self.weights = np.random.randn(input_size).astype(np.float32) * 0.01
        self.bias = np.float32(0.0)
        self.losses = []

    def sigmoid(self, z):
        # Clip z to prevent overflow
        z_clipped = self.xp.clip(z, -500, 500)
        return 1 / (1 + self.xp.exp(-z_clipped))

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        linear_output = self.xp.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_output)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)

    def compute_loss(self, y_true, y_pred):
        """Binary cross-entropy loss for binary classification."""
        y_pred_clipped = self.xp.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = -self.xp.mean(
            y_true * self.xp.log(y_pred_clipped) +
            (1 - y_true) * self.xp.log(1 - y_pred_clipped)
        )
        return float(loss)

    def fit(self, X, y):
        # Convert inputs to numpy arrays
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        print(f"Training SLP with {X.shape[0]} samples...")
        
        for i in range(self.num_iterations):
            # Forward pass
            probabilities = self.predict_proba(X)
            loss = self.compute_loss(y, probabilities)
            self.losses.append(loss)
            
            # Print progress
            if (i + 1) % 100 == 0 or i == 0:
                print(f"Iteration {i+1}/{self.num_iterations}, Loss: {loss:.6f}")

            # Backward pass (gradient descent)
            error = probabilities - y
            gradient = self.xp.dot(X.T, error) / len(y)
            bias_gradient = self.xp.mean(error)

            # Update parameters
            self.weights -= self.learning_rate * gradient
            self.bias -= self.learning_rate * bias_gradient

        return self.weights, float(self.bias), self.losses