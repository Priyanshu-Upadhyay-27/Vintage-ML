import numpy as np

class GradientDescent:
    def __init__(self, lr=0.01, ep=1000):
        self.m = np.random.randn()
        self.b = np.random.randn()
        self.learning_rate = lr
        self.epochs = ep
        self.loss_history = []

    def fit(self, X, y):
        X = X.flatten()
        y = y.flatten()
        N = len(X)

        for epoch in range(self.epochs):
            y_pred = self.m * X + self.b

            # The Backward Pass (Gradients)
            # Notice the np.sum() which replaces your big Sigma symbol.
            # We divide by N to get the Mean Squared Error gradient, keeping it stable.
            dm = (-2 / N) * np.sum(X * (y - y_pred))
            db = (-2 / N) * np.sum(y - y_pred)

            # STEP 3: The Parameter Update (Taking a step down the hill)
            self.m = self.m - (self.learning_rate * dm)
            self.b = self.b - (self.learning_rate * db)

            # Optional: Calculate and save the current loss to see if it's going down
            current_loss = np.mean((y - y_pred) ** 2)
            self.loss_history.append(current_loss)

