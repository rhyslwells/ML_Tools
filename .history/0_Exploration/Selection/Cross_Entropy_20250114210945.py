import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

# Load the Iris dataset
data = load_iris()
X = data['data']
y = data['target']
classes = data['target_names']

# One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Train a logistic regression model
model = LogisticRegression(max_iter=200, multi_class='multinomial')
model.fit(X_train, np.argmax(y_train, axis=1))

# Get predictions on the test set
predicted_probs = model.predict_proba(X_test)

# Function to calculate cross entropy loss for multiple predictions
def cross_entropy_loss(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return -np.sum(y_true * np.log(y_pred + 1e-15), axis=1)  # Adding epsilon for numerical stability

# Function to calculate mean squared error for multiple predictions
def mean_squared_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2, axis=1)

# Calculate cross entropy loss and mean squared error for all test samples
cross_entropy_losses = cross_entropy_loss(y_test, predicted_probs)
mse_losses = mean_squared_error(y_test, predicted_probs)

# Visualize the distributions
plt.figure(figsize=(12, 6))

# Cross Entropy Loss Distribution
plt.subplot(1, 2, 1)
plt.hist(cross_entropy_losses, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Cross Entropy Loss Distribution')
plt.xlabel('Loss')
plt.ylabel('Frequency')
plt.grid(True)

# Mean Squared Error Distribution
plt.subplot(1, 2, 2)
plt.hist(mse_losses, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
plt.title('Mean Squared Error Distribution')
plt.xlabel('Loss')
plt.ylabel('Frequency')
plt.grid(True)

plt.tight_layout()
plt.show()

# Display some statistics about the losses
print("Cross Entropy Loss Statistics:")
print(f"Mean: {np.mean(cross_entropy_losses):.4f}")
print(f"Median: {np.median(cross_entropy_losses):.4f}")
print(f"Max: {np.max(cross_entropy_losses):.4f}")
print(f"Min: {np.min(cross_entropy_losses):.4f}")

print("\nMean Squared Error Statistics:")
print(f"Mean: {np.mean(mse_losses):.4f}")
print(f"Median: {np.median(mse_losses):.4f}")
print(f"Max: {np.max(mse_losses):.4f}")
print(f"Min: {np.min(mse_losses):.4f}")

# Analyze the comparison between the two losses
def analyze_loss_comparison(cross_entropy, mse):
    correlation = np.corrcoef(cross_entropy, mse)[0, 1]
    print("\nAnalysis of Loss Comparison:")
    print(f"Correlation between Cross Entropy and MSE: {correlation:.4f}")
    
    if correlation > 0.7:
        print("The losses are strongly correlated, indicating similar patterns in penalizing predictions.")
    elif correlation > 0.3:
        print("The losses are moderately correlated, suggesting some differences in how they penalize errors.")
    else:
        print("The losses are weakly correlated, reflecting significant differences in their behavior.")

# Perform the analysis
analyze_loss_comparison(cross_entropy_losses, mse_losses)
