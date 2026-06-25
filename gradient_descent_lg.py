import numpy as np

class GradientDescent:
    def __init__(self, lr=0.01, ep=1000):
        self.m = np.random.randn()
        self.b = np.random.randn()
        self.learning_rate = lr
        self.epochs = ep
        self.loss_history = []

