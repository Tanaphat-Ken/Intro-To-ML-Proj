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

class MLPFromScratch:
    def __init__(self, layer_sizes, learning_rate=0.01, num_iterations=1000, use_cuda=True):
        self.layer_sizes = layer_sizes
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
        
        self.weights = []
        self.biases = []
        self.losses = []

        # Initialize weights and biases for each layer
        # Using Xavier initialization for better learning
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization: sqrt(1/n_in) for sigmoid activation
            scale = np.sqrt(1.0 / layer_sizes[i])
            if self.use_cuda:
                weight_matrix = cp.random.randn(layer_sizes[i], layer_sizes[i + 1]).astype(cp.float32) * scale
                bias_vector = cp.zeros((1, layer_sizes[i + 1]), dtype=cp.float32)
            else:
                weight_matrix = np.random.randn(layer_sizes[i], layer_sizes[i + 1]).astype(np.float32) * scale
                bias_vector = np.zeros((1, layer_sizes[i + 1]), dtype=np.float32)
            
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def sigmoid(self, z):
        # Clip z to prevent overflow
        z_clipped = self.xp.clip(z, -500, 500)
        return 1 / (1 + self.xp.exp(-z_clipped))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def predict_proba(self, X):
        # Ensure X is on the same device (GPU/CPU) as weights
        if self.use_cuda and not isinstance(X, cp.ndarray):
            X = cp.asarray(X, dtype=cp.float32)
        elif not self.use_cuda and isinstance(X, cp.ndarray):
            X = cp.asnumpy(X).astype(np.float32)
        
        a = X
        for w, b in zip(self.weights, self.biases):
            z = self.xp.dot(a, w) + b
            a = self.sigmoid(z)
        return a

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
        
        # Ensure y is 2D for matrix operations
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        print(f"Training on {'GPU' if self.use_cuda else 'CPU'} with {X.shape[0]} samples...")
        
        for iteration in range(self.num_iterations):
            # Forward pass
            activations = [X]
            a = X
            for w, b in zip(self.weights, self.biases):
                z = self.xp.dot(a, w) + b
                a = self.sigmoid(z)
                activations.append(a)

            loss = self.compute_loss(y, a)
            self.losses.append(loss)
            
            # Print progress
            if (iteration + 1) % 200 == 0 or iteration == 0:
                print(f"Iteration {iteration+1}/{self.num_iterations}, Loss: {loss:.6f}")

            # Backward pass (backpropagation)
            delta = (a - y) * self.sigmoid_derivative(a)
            for i in reversed(range(len(self.weights))):
                a_prev = activations[i]
                dw = self.xp.dot(a_prev.T, delta) / y.size
                db = self.xp.sum(delta, axis=0, keepdims=True) / y.size

                if i != 0:
                    delta = self.xp.dot(delta, self.weights[i].T) * self.sigmoid_derivative(activations[i])
                
                self.weights[i] -= self.learning_rate * dw
                self.biases[i] -= self.learning_rate * db
        
        # Convert final weights and biases back to NumPy for compatibility
        if self.use_cuda:
            final_weights = [cp.asnumpy(w) for w in self.weights]
            final_biases = [cp.asnumpy(b) for b in self.biases]
        else:
            final_weights = self.weights
            final_biases = self.biases
        
        return final_weights, final_biases, self.losses