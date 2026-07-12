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

