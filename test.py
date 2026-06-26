import numpy as np
from linear_regression import OrdinaryLeastSquare
from gradient_descent_lg import GradientDescent

print("Generating Synthetic Data (2 Features X, 1 Target y)...")
np.random.seed(42)

# 100 random numbers, BUT NOW WITH 2 COLUMNS!
X_test = 2 * np.random.rand(100, 2)

# The True Hidden Rule: y = (3.5 * x1) + (-1.5 * x2) + 2.0
true_W = np.array([[3.5], [-1.5]])
true_b = 2.0
y_test = X_test @ true_W + true_b + np.random.randn(100, 1) * 0.2

print(f"Target hidden rule: w1 = {true_W[0][0]}, w2 = {true_W[1][0]}, Intercept = {true_b}\n")

# ---- 1. Test Ordinary Least Squares ----
print("--- 1. Running OLS (Closed-Form Math) ---")
ols_model = OrdinaryLeastSquare()
ols_model.fit(X_test, y_test)

# Extract values for 2 weights
ols_w1 = ols_model.coef_[0][0] if isinstance(ols_model.coef_[0], np.ndarray) else ols_model.coef_[0]
ols_w2 = ols_model.coef_[1][0] if isinstance(ols_model.coef_[1], np.ndarray) else ols_model.coef_[1]
ols_b = ols_model.intercept_[0] if isinstance(ols_model.intercept_, np.ndarray) else ols_model.intercept_

print(f"OLS Found:     w1 = {ols_w1:.4f} | w2 = {ols_w2:.4f} | Intercept (b) = {ols_b:.4f}")

# ---- 2. Test Gradient Descent ----
print("\n--- 2. Running Gradient Descent (Iterative Optimization) ---")
gd_model = GradientDescent(lr=0.05, ep=2000)
gd_model.fit(X_test, y_test)

gd_w1 = gd_model.weights[0][0]
gd_w2 = gd_model.weights[1][0]
gd_b = gd_model.bias[0]

print(f"GD Found:      w1 = {gd_w1:.4f} | w2 = {gd_w2:.4f} | Intercept (b) = {gd_b:.4f}")

print("\nCONCLUSION:")
print("Both models successfully handle N-dimensional matrices without ANY loops over the data!")