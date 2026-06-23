# Linear Regression - OLS method: Ordinary Least Square
# good for 2D and similar dimension, not suitable for large dimenesion
import numpy as np

class LinearRegression:
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


