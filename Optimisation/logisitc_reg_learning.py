'''
### Optimizing Gradient Descent with Momentum and Adam

In this guide, we explore how to implement and compare two optimization techniques—Momentum and Adam—using gradient descent. 
Both methods are designed to improve the efficiency and performance of the learning process, particularly when training machine learning models.

#### Step 1: Dataset Preparation

We begin by generating a toy dataset using `sklearn.datasets.make_classification` and splitting it into training and validation sets using `train_test_split`. 
This dataset consists of two features and two classes, with 1000 samples.

#### Step 2: Sigmoid Activation Function

We use the sigmoid function to compute probabilities for binary classification. The function computes the logistic sigmoid of a given input.

- **Function**: `sigmoid(z)`

#### Step 3: Cost Function with L2 Regularization

We define the cost function that computes the binary cross-entropy loss, along with an L2 regularization term to prevent overfitting.

- **Function**: `cost_function(X, y, theta, lambda_reg=1.0)`

#### Step 4: Gradient Descent with Momentum

Momentum-based gradient descent accelerates the learning process by using the previous gradients to update parameters. This method reduces oscillations and helps the algorithm converge faster.

- **Function**: `gradient_descent_momentum(X, y, theta, alpha, iterations, lambda_reg=1.0, beta=0.9, batch_size=32, patience=100)`

**Challenges and Suggestions**:
- **Parameter Initialization**: Initialize `theta` with small random values to avoid poor convergence.
- **Learning Rate (`alpha`)**: Start with a small learning rate and adjust if necessary.
- **Patience**: Use early stopping to prevent overfitting and unnecessary computation.

#### Step 5: Adam Optimizer

The Adam optimizer combines momentum and adaptive learning rates for each parameter, providing better convergence for large datasets or complex models.

- **Function**: `adam_optimizer(X, y, theta, alpha, iterations, lambda_reg=1.0, beta1=0.9, beta2=0.999, epsilon=1e-8, batch_size=32, patience=100)`

**Challenges and Suggestions**:
- **Parameter Tuning**: Tune the learning rate (`alpha`), `beta1`, and `beta2` for optimal results.
- **Numerical Stability**: The epsilon parameter prevents division by zero. Adjust as needed to avoid numeric errors.

#### Step 6: Experimenting with Hyperparameters

Experiment with different combinations of hyperparameters, such as learning rate, beta, and lambda regularization, using grid search or random search to optimize model performance.

#### Step 7: Evaluating Results

Evaluate model performance using accuracy and precision metrics on the validation set. Record the results for comparison across different optimizers and hyperparameter configurations.

#### Step 8: Visualizing the Cost Convergence

Visualize the convergence of the cost function over iterations for both the momentum and Adam optimizers. This provides insight into how quickly each optimizer converges and their efficiency in training.

- **Function**: `plt.plot(...)` for cost visualization

### Conclusion

By following this structured approach, you can implement and compare the Momentum and Adam optimizers to improve training efficiency and accuracy. 
Experimenting with various hyperparameters and using early stopping ensures that models converge quickly without overfitting.
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import pandas as pd

# Generate a toy dataset
np.random.seed(42)  # For reproducibility
X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, n_informative=2, n_redundant=0, random_state=42)
y = y.reshape(-1)  # Flatten for sklearn compatibility

# Split into train and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Sigmoid function for manual probability computation with numerical stability
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip z to avoid overflow

# Cost function (manual calculation for gradient descent with L2 regularization)
def cost_function(X, y, theta, lambda_reg=1.0):
    m = len(y)
    z = np.dot(np.hstack((np.ones((X.shape[0], 1)), X)), theta)  # Add bias term
    h = sigmoid(z)
    cost = -(1 / m) * (np.dot(y, np.log(h)) + np.dot((1 - y), np.log(1 - h)))
    # Add regularization term (L2)
    reg_cost = (lambda_reg / (2 * m)) * np.sum(np.square(theta[1:]))  # Regularization, excluding intercept
    return cost + reg_cost

# Gradient Descent with Mini-batch and Early Stopping
def gradient_descent_momentum(X, y, theta, alpha, iterations, lambda_reg=1.0, beta=0.9, batch_size=32, patience=100):
    m = len(y)
    v = np.zeros_like(theta)  # Initialize velocity
    costs = []
    best_cost = float('inf')
    no_improvement_count = 0
    early_stopping_iteration = None
    
    # Loop for mini-batch gradient descent
    for i in range(iterations):
        # Shuffle the data at the start of each epoch for mini-batch gradient descent
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for j in range(0, m, batch_size):
            X_batch = X_shuffled[j:j+batch_size]
            y_batch = y_shuffled[j:j+batch_size]
            
            z = np.dot(np.hstack((np.ones((X_batch.shape[0], 1)), X_batch)), theta)  # Add bias term
            h = sigmoid(z)
            gradient = (1 / len(y_batch)) * np.dot(np.hstack((np.ones((X_batch.shape[0], 1)), X_batch)).T, (h - y_batch))
            # Regularization term
            gradient[1:] += (lambda_reg / len(y_batch)) * theta[1:]  # Exclude intercept term from regularization
            
            # Update velocity and parameters
            v = beta * v + (1 - beta) * gradient
            theta -= alpha * v  # Update theta
        
        # Calculate cost after each epoch
        cost = cost_function(X, y, theta, lambda_reg)
        costs.append(cost)
        
        # Early stopping
        if cost < best_cost:
            best_cost = cost
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        if no_improvement_count >= patience:
            early_stopping_iteration = i + 1
            print(f"Early stopping at iteration {early_stopping_iteration}")
            break
    
    return theta, costs, early_stopping_iteration

# Adam Optimizer with Mini-batch and Early Stopping
def adam_optimizer(X, y, theta, alpha, iterations, lambda_reg=1.0, beta1=0.9, beta2=0.999, epsilon=1e-8, batch_size=32, patience=100):
    m = len(y)
    m_t = np.zeros_like(theta)
    v_t = np.zeros_like(theta)
    costs = []
    best_cost = float('inf')
    no_improvement_count = 0
    early_stopping_iteration = None
    
    for t in range(1, iterations + 1):
        # Shuffle the data at the start of each epoch for mini-batch gradient descent
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for j in range(0, m, batch_size):
            X_batch = X_shuffled[j:j+batch_size]
            y_batch = y_shuffled[j:j+batch_size]
            
            z = np.dot(np.hstack((np.ones((X_batch.shape[0], 1)), X_batch)), theta)  # Add bias term
            h = sigmoid(z)
            gradient = (1 / len(y_batch)) * np.dot(np.hstack((np.ones((X_batch.shape[0], 1)), X_batch)).T, (h - y_batch))
            # Regularization term
            gradient[1:] += (lambda_reg / len(y_batch)) * theta[1:]  # Exclude intercept term from regularization
            
            # Update first and second moment estimates
            m_t = beta1 * m_t + (1 - beta1) * gradient
            v_t = beta2 * v_t + (1 - beta2) * (gradient ** 2)
            
            # Bias correction
            m_t_hat = m_t / (1 - beta1 ** t)
            v_t_hat = v_t / (1 - beta2 ** t)
            
            # Update parameters
            theta -= alpha * m_t_hat / (np.sqrt(v_t_hat) + epsilon)
        
        # Calculate cost after each epoch
        cost = cost_function(X, y, theta, lambda_reg)
        costs.append(cost)
        
        # Early stopping
        if cost < best_cost:
            best_cost = cost
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        if no_improvement_count >= patience:
            early_stopping_iteration = t
            print(f"Early stopping at iteration {early_stopping_iteration}")
            break
    
    return theta, costs, early_stopping_iteration

# Hyperparameter grid
alpha_values = [0.001, 0.005, 0.01]
beta_values = [0.9, 0.99]
lambda_values = [0.1, 1.0, 10.0]

# Store the results
results = []

# Experiment with different hyperparameters
for alpha in alpha_values:
    for beta in beta_values:
        for lambda_reg in lambda_values:
            # Momentum
            theta_init = np.zeros(X_train.shape[1] + 1)  # +1 for the intercept (bias) term
            theta_momentum, costs_momentum, momentum_early_stopping = gradient_descent_momentum(X_train, y_train, theta_init, alpha, 10000, lambda_reg, beta)
            
            # Adam
            theta_adam, costs_adam, adam_early_stopping = adam_optimizer(X_train, y_train, theta_init, alpha, 10000, lambda_reg)
            
            # Evaluate on validation set
            z_momentum = np.dot(np.hstack((np.ones((X_val.shape[0], 1)), X_val)), theta_momentum)
            h_momentum = sigmoid(z_momentum)
            predictions_momentum = (h_momentum >= 0.5).astype(int)
            
            z_adam = np.dot(np.hstack((np.ones((X_val.shape[0], 1)), X_val)), theta_adam)
            h_adam = sigmoid(z_adam)
            predictions_adam = (h_adam >= 0.5).astype(int)
            
            # Calculate accuracy and precision on validation set
            accuracy_momentum = accuracy_score(y_val, predictions_momentum)
            precision_momentum = precision_score(y_val, predictions_momentum)
            
            accuracy_adam = accuracy_score(y_val, predictions_adam)
            precision_adam = precision_score(y_val, predictions_adam)
            
            # Store the results with early stopping information
            results.append([alpha, beta, lambda_reg, accuracy_momentum, precision_momentum, accuracy_adam, precision_adam,
                            momentum_early_stopping, adam_early_stopping])

# Convert results to a pandas DataFrame
df_results = pd.DataFrame(results, columns=['Learning Rate', 'Momentum Beta', 'Lambda', 
                                             'Momentum Accuracy', 'Momentum Precision', 
                                             'Adam Accuracy', 'Adam Precision', 
                                             'Momentum Early Stopping', 'Adam Early Stopping'])

# Display the table of results
print("\nComparison of Momentum and Adam Performance with Early Stopping Iteration:")
print(df_results[8:9])

# Best configuration for Momentum and Adam
best_momentum = df_results.loc[df_results['Momentum Accuracy'].idxmax()]
best_adam = df_results.loc[df_results['Adam Accuracy'].idxmax()]

print(f"\nBest Momentum: {best_momentum}")
print(f"Best Adam: {best_adam}")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(np.array(range(len(costs_momentum))), costs_momentum, label=f"Momentum (beta={best_momentum['Momentum Beta']})", color='blue')
plt.plot(np.array(range(len(costs_adam))), costs_adam, label=f"Adam", color='red')
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Comparison: Momentum vs Adam (Best Configurations)")
plt.legend()
plt.show()
