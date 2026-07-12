import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        """
        Math: y_hat = 1 / (1 + e^(-z))
        Squashes our linear output strictly between 0 and 1 to give a probability.
        """
        # np.clip prevents overflow warnings if z gets too massive
        z = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        # Ensure y is a 2D column vector (N, 1) to match matrix math shapes
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        N, D = X.shape  # N = number of rows (samples), D = number of features

        # Step 1 Initialization: Start with random weights (or zeros)
        self.weights = np.zeros((D, 1))
        self.bias = 0.0

        for epoch in range(self.epochs):
            # ---------------------------------------------------------
            # STEP 1 & 2: The Forward Pass (Hypothesis)
            # ---------------------------------------------------------

            # Math: z = w1*x1 + w2*x2 + ... + b
            z = X @ self.weights + self.bias

            # Math: y_hat = sigmoid(z)
            y_hat = self._sigmoid(z)

            # ---------------------------------------------------------
            # TRACKING THE LOSS (Optional, just to watch the bowl go down)
            # ---------------------------------------------------------
            # Math: J = - [y * log(y_hat) + (1-y) * log(1 - y_hat)]
            # We add a tiny epsilon (1e-15) so we don't accidentally do log(0)
            epsilon = 1e-15
            y_hat_safe = np.clip(y_hat, epsilon, 1 - epsilon)
            loss = -(1 / N) * np.sum(y * np.log(y_hat_safe) + (1 - y) * np.log(1 - y_hat_safe))
            self.loss_history.append(loss)

            # ---------------------------------------------------------
            # STEP 3: The Gradients (The Magic Cancellation)
            # ---------------------------------------------------------
            # Because the calculus canceled out the nasty log/sigmoid derivatives,
            # our error is simply: (Prediction - Actual)
            error = y_hat - y

            # Math: dw = (y_hat - y) * x
            # In matrix form, we use X.T (Transpose) to multiply every feature 
            # column by the error column simultaneously, then divide by N for the average.
            dw = (1 / N) * (X.T @ error)

            # For bias, we just sum up the errors and average them.
            db = (1 / N) * np.sum(error)

            # ---------------------------------------------------------
            # STEP 4: The Update Rule
            # ---------------------------------------------------------
            # Math: w_new = w_old - (learning_rate * dw)
            self.weights = self.weights - (self.lr * dw)
            self.bias = self.bias - (self.lr * db)

    def predict_proba(self, X):
        """Returns the raw probability (e.g., 0.85)"""
        z = X @ self.weights + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        """Returns the hard class (1 or 0) based on the 0.5 threshold"""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

# Now we will also make the perceptron too.
