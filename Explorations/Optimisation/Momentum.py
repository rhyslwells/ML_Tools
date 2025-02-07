import numpy as np
import matplotlib.pyplot as plt

def gradient_descent_with_momentum(beta=0.9, learning_rate=0.1, max_iter=100, tolerance=1e-6):
    """
    Gradient descent with momentum.

    Parameters:
    beta (float): Momentum coefficient.
    learning_rate (float): Learning rate for gradient descent.
    max_iter (int): Maximum number of iterations.
    tolerance (float): Convergence tolerance.

    Returns:
    list: History of the cost function values.
    """
    # Define a more complex cost function: J(theta) = theta^4 - 3*theta^3 + 2
    def cost_function(theta):
        return theta ** 4 - 3 * theta ** 3 + 2

    def gradient(theta):
        return 4 * theta ** 3 - 9 * theta ** 2

    # Initialization
    theta = 2.5  # Starting point
    v = 0  # Initial velocity
    cost_history = []

    for _ in range(max_iter):
        grad = gradient(theta)
        v = beta * v + (1 - beta) * grad  # Update velocity
        theta -= learning_rate * v  # Update parameter
        cost = cost_function(theta)
        cost_history.append(cost)

        # Check for convergence
        if abs(grad) < tolerance:
            break

    return cost_history

def standard_gradient_descent(learning_rate=0.1, max_iter=100, tolerance=1e-6):
    """
    Standard gradient descent without momentum.

    Parameters:
    learning_rate (float): Learning rate for gradient descent.
    max_iter (int): Maximum number of iterations.
    tolerance (float): Convergence tolerance.

    Returns:
    list: History of the cost function values.
    """
    # Define a more complex cost function: J(theta) = theta^4 - 3*theta^3 + 2
    def cost_function(theta):
        return theta ** 4 - 3 * theta ** 3 + 2

    def gradient(theta):
        return 4 * theta ** 3 - 9 * theta ** 2

    # Initialization
    theta = 2.5  # Starting point
    cost_history = []

    for _ in range(max_iter):
        grad = gradient(theta)
        theta -= learning_rate * grad  # Update parameter
        cost = cost_function(theta)
        cost_history.append(cost)

        # Check for convergence
        if abs(grad) < tolerance:
            break

    return cost_history

# Visualization of the results
betas = [0.5, 0.9, 0.99]
learning_rate = 0.1
max_iter = 100

plt.figure(figsize=(12, 8))

# Standard Gradient Descent
cost_no_momentum = standard_gradient_descent(learning_rate=learning_rate, max_iter=max_iter)
plt.plot(cost_no_momentum, label="No Momentum", linestyle="--")

# Gradient Descent with Momentum
for beta in betas:
    cost_with_momentum = gradient_descent_with_momentum(beta=beta, learning_rate=learning_rate, max_iter=max_iter)
    plt.plot(cost_with_momentum, label=f"Momentum (beta={beta})")

plt.title("Gradient Descent: Momentum vs No Momentum (Complex Cost Function)")
plt.xlabel("Iterations")
plt.ylabel("Cost Function Value")
plt.legend()
plt.grid()
plt.show()
