import numpy as np

class GradientDescent:
    def __init__(self, lr=0.01, ep=1000):
        # We don't initialize weights here anymore because we don't know the dimensions yet!
        self.weights = None
        self.bias = None
        self.learning_rate = lr
        self.epochs = ep
        self.loss_history = []

    def fit(self, X, y):
        # Safety check: ensure y is a 2D column vector (N, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # N = number of rows (samples), D = number of columns (features)
        N, D = X.shape

        # Initialize weights to shape (D, 1) and bias to a single float
        self.weights = np.random.randn(D, 1)
        self.bias = np.random.randn(1)

        for epoch in range(self.epochs):
            # Forward Pass: Matrix multiplication (N, D) @ (D, 1) = (N, 1)
            y_pred = X @ self.weights + self.bias

            # Calculate Error
            error = y_pred - y

            # Backward Pass: Transpose X to align dimensions for the gradient dot product
            # (D, N) @ (N, 1) = (D, 1) -> Exactly the shape of our weights!
            dw = (2 / N) * (X.T @ error)
            db = (2 / N) * np.sum(error)

            # Parameter Update
            self.weights = self.weights - (self.learning_rate * dw)
            self.bias = self.bias - (self.learning_rate * db)

            # Track loss
            current_loss = np.mean(error ** 2)
            self.loss_history.append(current_loss)

    def predict(self, X):
        # Automatically handles however many columns X has
        return X @ self.weights + self.bias
