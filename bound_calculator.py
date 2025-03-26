import numpy as np
import torch
from scipy.linalg import svd

# Step 1: Define Network Structure
layers = [2, 32, 32, 4]  # 2 input neurons, two hidden layers with 32 neurons, 4 output neurons

# Step 2: Generate random weight matrices
torch.manual_seed(0)  # For reproducibility
weights = [torch.randn(layers[i+1], layers[i]) * 0.1 for i in range(len(layers)-1)]

# Step 3: Compute Spectral Norm Complexity
def spectral_complexity(weights, layers, gamma_margin=1.0):
    """
    Computes the spectral complexity measure:
    (1 / gamma_margin^2) * Π ||W_l||_2^2 * h_l
    """
    spectral_norms = [np.linalg.norm(svd(W.detach().numpy(), compute_uv=False), ord=2) for W in weights]
    complexity = np.prod(spectral_norms) * np.prod(layers[1:]) / (gamma_margin ** 2)
    return complexity

# Compute Complexity
complexity = spectral_complexity(weights, layers)
print("Spectral Norm Complexity:", complexity)

# Step 4: Compute Prior P(c) using Exponential Distribution
lambda_val = 1e-4  # Adjust lambda to prevent numerical instability
P_c = lambda_val * np.exp(-lambda_val * complexity)

# Check results
print("P(c):", P_c)

# Step 5: Compute Bound using Chernoff Relaxation
m = 100  # Number of training samples
delta = 0.05  # Confidence level
c_S_hat = np.arange(0, m + 1)  # Empirical error values

chernoff_bound_upper = c_S_hat / m + np.sqrt((np.log(1/P_c) + np.log(1/delta)) / (2*m))
chernoff_bound_lower = c_S_hat / m - np.sqrt((np.log(1/P_c) + np.log(1/delta)) / (2*m))

# Step 6: Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(c_S_hat / m, chernoff_bound_upper, label="Chernoff Upper Bound", linestyle='--', color='red')
plt.plot(c_S_hat / m, chernoff_bound_lower, label="Chernoff Lower Bound", linestyle='--', color='blue')

plt.xlabel("Empirical Error Rate (c_S_hat / m)")
plt.ylabel("Error Bound")
plt.title("Chernoff Relaxation Bound for Neural Network")
plt.legend()
plt.grid(True)
plt.show()

# Output the computed values
print(f"Spectral Complexity: {complexity}")
print(f"P(c): {P_c}")
print(f"Sum of P(c) (should be 1): {np.sum(P_c)}")
