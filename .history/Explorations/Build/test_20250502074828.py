"""
# 📘 Introduction

This script demonstrates the phenomenon of **vanishing** and **exploding gradients** in deep neural networks.
We simulate a forward and backward pass through a deep stack of layers and observe how the gradient magnitude behaves.

We'll explore:
- ReLU and Sigmoid activations
- The effect of initial weight scaling
- How gradients can shrink or blow up across layers

This is a synthetic example for intuition building.
"""

import numpy as np
import matplotlib.pyplot as plt

# Seed for reproducibility
np.random.seed(42)

"""
# 🔧 Step 1: Configuration

- Number of layers: depth of the network
- Number of neurons per layer
- Activation function: sigmoid or ReLU
- Weight initialization scale

We'll run experiments with both small and large initializations.
"""

depth = 50
layer_width = 100
activation_type = 'sigmoid'  # or 'relu'
weight_scale = 1.0  # try 0.5 for vanishing, 2.0 for exploding

"""
# 🧠 Step 2: Activation Functions
"""

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

activation = sigmoid if activation_type == 'sigmoid' else relu
activation_derivative = sigmoid_derivative if activation_type == 'sigmoid' else relu_derivative

"""
# 🔁 Step 3: Forward Pass and Backpropagation Simulation

We simulate:

- a forward pass where input is propagated through many layers
- a backward pass where gradients are multiplied through Jacobians

We track the norm of the gradient at each layer.
"""

# Initialize input and gradient
x = np.random.randn(layer_width)
gradient = np.ones_like(x)

# Store gradient norms
forward_norms = []
backward_norms = []

# Forward pass
for _ in range(depth):
    W = weight_scale * np.random.randn(layer_width, layer_width) / np.sqrt(layer_width)
    x = activation(W @ x)
    forward_norms.append(np.linalg.norm(x))

# Reset x and simulate backpropagation
grad = np.ones(layer_width)
x = np.random.randn(layer_width)

for _ in range(depth):
    W = weight_scale * np.random.randn(layer_width, layer_width) / np.sqrt(layer_width)
    z = W @ x
    x = activation(z)
    grad = W.T @ (grad * activation_derivative(z))
    backward_norms.append(np.linalg.norm(grad))

"""
# 📈 Step 4: Visualization

We plot how the forward activations and backward gradient norms evolve through depth.
This shows clearly the vanishing or exploding behavior.
"""

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(forward_norms)
plt.title('Forward Pass Activation Norms')
plt.xlabel('Layer')
plt.ylabel('Norm')

plt.subplot(1, 2, 2)
plt.plot(backward_norms)
plt.title('Backward Pass Gradient Norms')
plt.xlabel('Layer')
plt.ylabel('Norm')

plt.tight_layout()
plt.show()

"""
# ✅ Observations

Try changing:
- `activation_type` between `'sigmoid'` and `'relu'`
- `weight_scale` between `0.5`, `1.0`, and `2.0`

You will observe:
- With sigmoid and small weights: **vanishing gradients**
- With large weights: **exploding gradients**
- ReLU partially mitigates vanishing but may still explode if weights are too large

This highlights why modern initialization schemes (e.g., He, Xavier) and architectures (e.g., residual connections) are important.
"""
