# Linear Regression - OLS method: Ordinary Least Square
# good for 2D and similar dimension, not suitable for large dimension
import numpy as np

class OrdinaryLeastSquare:
    def __init__(self):
        self.weights = None
        self.intercepts_ = None
        self.coef_ = None

    def _pad_with_ones(self, X: np.ndarray) -> np.ndarray:
        """
        STEP 1 (The Helper):
        Take an incoming matrix X of shape (N, D).
        You need to return a new matrix of shape (N, D + 1) where the very first column is all 1.0s.

        Hint: Look up `np.ones()` to make the column, and `np.column_stack()` to glue them together.
        """
        n_samples = X.shape[0]
        ones_column = np.ones((n_samples, 1))
        return np.column_stack((ones_column, X))


    def fit(self, X: np.ndarray, y: np.ndarray) -> "OrdinaryLeastSquare":
        # 1. Protect against 1D arrays
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # 2. Upgrade X to the Design Matrix (X)
        X_design = self._pad_with_ones(X)

        # 3. Execute the Normal Equation: Beta = (X^T * X)^(-1) * X^T * y
        X_T = X_design.T

        # (X^T @ X)
        covariance_matrix = X_T @ X_design

        # Inverse of covariance matrix
        # Note: We use np.linalg.pinv (pseudo-inverse) rather than .inv()
        # safely handling singular/collinear matrices without crashing Python.
        cov_inverse = np.linalg.pinv(covariance_matrix)

        # Complete the right side of the projection
        self.weights = cov_inverse @ X_T @ y

        # 4. Break out the friendly user properties
        self.intercept_ = self.weights[0]
        self.coef_ = self.weights[1:]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model is untrained. Call .fit() first.")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_design = self._pad_with_ones(X)
        return X_design @ self.weights

    def r2_score(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """Evaluates the model using the Coefficient of Determination (R^2)"""
        y_pred = self.predict(X)
        residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)
        total_sum_of_squares = np.sum((y_true - np.mean(y_true)) ** 2)

        return 1.0 - (residual_sum_of_squares / total_sum_of_squares)
