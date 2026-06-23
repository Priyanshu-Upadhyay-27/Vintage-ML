# Linear Regression - OLS method: Ordinary Least Square
# good for 2D and similar dimension, not suitable for large dimenesion
import numpy as np

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.intercepts_ = None
        self.coef_ = None


