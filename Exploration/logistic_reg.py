'''
### Summary of the Note

In this note, we compare the performance of logistic regression models fitted using **sklearn's `LogisticRegression`** and **gradient descent** with regularization. The objective is to show how gradient descent can optimize the model parameters (intercept and coefficients) to match those obtained from sklearn, which uses a more sophisticated optimization algorithm like **LBFGS**.

We demonstrate the following key points:

#### 1. **Parameter Comparison (Theta Values)**:
   In this note, we compare the **intercept** and **coefficients** from both the gradient descent and sklearn models. We directly compare the parameters using the Euclidean distance between the two sets of values. This comparison reveals how closely the gradient descent optimization matches the sklearn solution.

#### 2. **Convergence of Cost Function**:
   We track the cost function during gradient descent and compare it with the cost obtained from sklearn’s logistic regression. The goal is to show whether gradient descent converges to the same final cost as sklearn. If the gradient descent model converges properly, the final cost should be similar to the one obtained from sklearn.

#### 3. **Varying Learning Rate and Regularization Strength**:
   We experiment with different values of **learning rate (alpha)** and **regularization parameter (lambda_reg)**, creating a grid of configurations. The performance of the gradient descent model is evaluated for each combination. We show how adjusting these parameters can affect the optimization process and help achieve parameter values that are closer to sklearn’s solution.

#### 4. **Visualization**:
   In this note, we also plot the **intercept** and **coefficients** for both methods to visually inspect the differences. This allows us to compare the magnitudes and directions of the parameters and see how well gradient descent approximates the sklearn model.

#### 5. **Final Model Comparison**:
   We identify the **best configuration** of learning rate and regularization strength that minimizes the distance between the parameters obtained from gradient descent and those from sklearn. The final comparison of parameters shows the closeness of the gradient descent solution to sklearn’s solution.

#### Observations and Troubleshooting:
   We show that the gradient descent optimization sometimes fails to converge to the correct parameters due to several factors:
   - **Learning Rate (Alpha)**: A learning rate that is too high causes the gradient descent to **overshoot** the optimal solution, while a rate that is too low leads to slow convergence.
   - **Regularization Strength (lambda_reg)**: If the regularization parameter is too large, it overly penalizes the coefficients, leading to smaller values that may not match the sklearn model. If it's too small, it doesn’t effectively regularize the model.
   - **Gradient Descent Convergence**: Gradient descent can struggle to converge when starting from zero, especially with high-dimensional data or insufficient iterations.
   - **Parameter Initialization**: Starting gradient descent from zeros can cause slow convergence. A small random initialization could speed up the process.

#### Recommendations for Improvement:
   We suggest the following approaches to improve the gradient descent performance:
   - **Tune Hyperparameters**: Experiment with a wider range of learning rates (e.g., 0.001, 0.005) and regularization strengths (e.g., 0.01, 0.5).
   - **Increase Iterations**: Gradient descent may require more iterations (e.g., 1000 or more) to converge properly.
   - **Use Random Initialization**: Instead of starting from zero, initialize the parameters with small random values to improve convergence.
   - **Numerical Stability**: Apply numerical techniques such as **clamping the sigmoid function** to avoid overflow during optimization.

#### Conclusion:
   In this note, we show that while gradient descent can converge to a solution close to sklearn’s logistic regression, it requires careful tuning of hyperparameters and optimization strategies. By adjusting the **learning rate**, **regularization strength**, and increasing **iterations**, we can achieve better convergence and parameter estimates. This process illustrates the challenges and solutions for aligning gradient descent results with sklearn’s advanced optimization techniques.
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import pandas as pd

# Generate a toy dataset
np.random.seed(42)  # For reproducibility
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_informative=2, n_redundant=0, random_state=42)
y = y.reshape(-1)  # Flatten for sklearn compatibility

# 1. Fit Logistic Regression using sklearn (with regularization)
model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, C=1.0)
model.fit(X, y)

# Retrieve model parameters (intercept and coefficients)
intercept_sklearn = model.intercept_[0]
coefficients_sklearn = model.coef_[0]
theta_sklearn = np.hstack((intercept_sklearn, coefficients_sklearn))  # Combine intercept and coefficients

print("Sklearn Parameters (Intercept, Coefficients):", theta_sklearn)

# 2. Sigmoid function for manual probability computation with numerical stability
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip z to avoid overflow

# 3. Cost function (manual calculation for gradient descent with L2 regularization)
def cost_function(X, y, theta, lambda_reg=0.1):
    m = len(y)
    z = np.dot(np.hstack((np.ones((X.shape[0], 1)), X)), theta)  # Add bias term
    h = sigmoid(z)
    cost = -(1 / m) * (np.dot(y, np.log(h)) + np.dot((1 - y), np.log(1 - h)))
    # Add regularization term (L2)
    reg_cost = (lambda_reg / (2 * m)) * np.sum(np.square(theta[1:]))  # Regularization, excluding intercept
    return cost + reg_cost

# 4. Gradient Descent Implementation with L2 Regularization
def gradient_descent(X, y, theta, alpha, iterations, lambda_reg=0.1):
    m = len(y)
    costs = []
    for _ in range(iterations):
        z = np.dot(np.hstack((np.ones((X.shape[0], 1)), X)), theta)  # Add bias term
        h = sigmoid(z)
        gradient = (1 / m) * np.dot(np.hstack((np.ones((X.shape[0], 1)), X)).T, (h - y))
        # Regularization term
        gradient[1:] += (lambda_reg / m) * theta[1:]  # Exclude intercept term from regularization
        theta -= alpha * gradient  # Update theta
        costs.append(cost_function(X, y, theta, lambda_reg))  # Track the cost
    return theta, costs

# Initialize parameters
theta_init = np.zeros(X.shape[1] + 1)  # +1 for the intercept (bias) term
iterations = 10000  # Number of gradient descent steps

# Set up different values for lambda and learning rate to experiment with
lambda_values = [0.01, 0.1, 1.0, 10.0]
alpha_values = [0.001, 0.005, 0.01, 0.1]

# Store the results
results = []

# Run experiments for different lambda and alpha values
for lambda_reg in lambda_values:
    for alpha in alpha_values:
        theta_gd, costs = gradient_descent(X, y, theta_init, alpha, iterations, lambda_reg)
        final_cost_gd = costs[-1]

        # Calculate Euclidean distance between sklearn parameters and gradient descent parameters
        distance = np.linalg.norm(theta_gd - theta_sklearn)

        # Store the results: lambda, alpha, final cost, intercept, coefficients, distance
        results.append([lambda_reg, alpha, final_cost_gd, theta_gd[0], theta_gd[1], theta_gd[2], distance])

# Convert results to a pandas DataFrame for better visualization
df_results = pd.DataFrame(results, columns=['Lambda', 'Learning Rate', 'Final Cost', 'Intercept', 'Coef_1', 'Coef_2', 'Distance'])

# Display the table of results
print("\nComparison of Different Lambda and Learning Rate Combinations:")
print(df_results)

# 5. Identify the best configuration (minimizing the distance)
best_params = df_results.loc[df_results['Distance'].idxmin()]
lambda_best = best_params['Lambda']
alpha_best = best_params['Learning Rate']
print(f"\nBest lambda: {lambda_best}, Best Learning Rate: {alpha_best}")

# Re-run Gradient Descent with the best lambda and alpha
theta_gd_best, costs_best = gradient_descent(X, y, theta_init, alpha_best, iterations, lambda_best)

# Print the comparison of parameters for the best configuration
print("\nComparison of Parameters (Best Configuration):")
print(f"Intercept (sklearn): {intercept_sklearn} vs Gradient Descent: {theta_gd_best[0]}")
print(f"Coefficients (sklearn): {coefficients_sklearn} vs Gradient Descent: {theta_gd_best[1:]}")

# Final cost comparison for the best configuration
final_cost_best = costs_best[-1]
print(f"\nFinal Cost (Best Configuration): {final_cost_best}")
