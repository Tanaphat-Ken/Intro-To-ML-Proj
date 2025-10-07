import numpy as np

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    # Test if CUDA is actually working
    try:
        cp.cuda.Device(0).compute_capability
        CUDA_AVAILABLE = True
        print("CUDA acceleration available with CuPy")
    except Exception:
        CUDA_AVAILABLE = False
        print("CuPy installed but CUDA not functional, using CPU-only NumPy")
except ImportError:
    CUDA_AVAILABLE = False
    print("CuPy not available, using CPU-only NumPy")

class SLPFromScratch:
    def __init__(self, input_size, learning_rate=0.01, num_iterations=1000, use_cuda=True):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.use_cuda = use_cuda and CUDA_AVAILABLE
        
        # Choose the appropriate library (CuPy for GPU, NumPy for CPU)
        if self.use_cuda:
            self.xp = cp
            print("Using GPU acceleration with CuPy")
        else:
            self.xp = np
            print("Using CPU with NumPy")
        
        # Initialize weights and bias
        if self.use_cuda:
            self.weights = cp.random.randn(input_size).astype(cp.float32) * 0.01
            self.bias = cp.float32(0.0)
        else:
            self.weights = np.random.randn(input_size).astype(np.float32) * 0.01
            self.bias = np.float32(0.0)
        
        self.losses = []

    def sigmoid(self, z):
        # Clip z to prevent overflow
        z_clipped = self.xp.clip(z, -500, 500)
        return 1 / (1 + self.xp.exp(-z_clipped))

    def predict_proba(self, X):
        # Ensure X is on the same device (GPU/CPU) as weights
        if self.use_cuda and not isinstance(X, cp.ndarray):
            X = cp.asarray(X, dtype=cp.float32)
        elif not self.use_cuda and isinstance(X, cp.ndarray):
            X = cp.asnumpy(X).astype(np.float32)
        
        linear_output = self.xp.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_output)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)

    def compute_loss(self, y_true, y_pred):
        loss = self.xp.mean((y_true - y_pred) ** 2)  # Mean Squared Error
        return float(loss)  # Convert to Python float for compatibility

    def fit(self, X, y):
        # Convert inputs to appropriate format and device
        if self.use_cuda:
            if not isinstance(X, cp.ndarray):
                X = cp.asarray(X, dtype=cp.float32)
            if not isinstance(y, cp.ndarray):
                y = cp.asarray(y, dtype=cp.float32)
        else:
            if isinstance(X, cp.ndarray):
                X = cp.asnumpy(X).astype(np.float32)
            else:
                X = X.astype(np.float32)
            if isinstance(y, cp.ndarray):
                y = cp.asnumpy(y).astype(np.float32)
            else:
                y = y.astype(np.float32)
        
        print(f"Training on {'GPU' if self.use_cuda else 'CPU'} with {X.shape[0]} samples...")
        
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
            gradient = self.xp.dot(X.T, error) / y.size
            bias_gradient = self.xp.mean(error)
            
            # Update parameters
            self.weights -= self.learning_rate * gradient
            self.bias -= self.learning_rate * bias_gradient
        
        # Convert final weights and bias back to NumPy for compatibility
        if self.use_cuda:
            final_weights = cp.asnumpy(self.weights)
            final_bias = float(cp.asnumpy(self.bias))
        else:
            final_weights = self.weights
            final_bias = float(self.bias)
        
        return final_weights, final_bias, self.losses