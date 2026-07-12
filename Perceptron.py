import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.errors_history = []

    def _step_function(self, z):
        """
        The Perceptron's Activation Function.
        Instead of a smooth probability (Sigmoid), it makes a hard binary choice.
        """
        return 1 if z >= 0 else 0

    def fit(self, X, y):
        # Flatten y to a 1D array so we can easily loop through it
        y = y.ravel()
        N, D = X.shape

        # Step 1: Initialize weights and bias to zeros
        self.weights = np.zeros(D)
        self.bias = 0.0

        for epoch in range(self.epochs):
            errors_in_epoch = 0

            # The classic Perceptron iterates row by row (Stochastic update)
            for i in range(N):
                x_i = X[i]
                y_true = y[i]

                # Step 2: Linear combination (z)
                z = np.dot(x_i, self.weights) + self.bias

                # Step 3: Activation (Hard guess: 0 or 1)
                y_hat = self._step_function(z)

                # Step 4: The Update Factor
                # If correct: y_true - y_hat = 0 (No update)
                # If false negative (Truth=1, Guess=0): 1 - 0 = +1 (Add weights)
                # If false positive (Truth=0, Guess=1): 0 - 1 = -1 (Subtract weights)
                update = self.lr * (y_true - y_hat)

                # Step 5: Update weights and bias (Only happens if update != 0)
                if update != 0:
                    self.weights += update * x_i
                    self.bias += update
                    errors_in_epoch += 1

            self.errors_history.append(errors_in_epoch)

            # Early stopping: if the line separates everything perfectly, stop!
            if errors_in_epoch == 0:
                print(f"Converged beautifully! Zero errors at epoch {epoch}.")
                break

    def predict(self, X):
        """Predicts the classes for a whole matrix X at once."""
        pass