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
encoder = OneHotEncoder(sparse=False)
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

# Calculate cross entropy loss for all test samples
losses = cross_entropy_loss(y_test, predicted_probs)

# Visualize the losses
plt.figure(figsize=(10, 6))
plt.hist(losses, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Distribution of Cross Entropy Loss Across Test Samples')
plt.xlabel('Loss')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Display some statistics about the losses
print(f"Mean Loss: {np.mean(losses):.4f}")
print(f"Median Loss: {np.median(losses):.4f}")
print(f"Max Loss: {np.max(losses):.4f}")
print(f"Min Loss: {np.min(losses):.4f}")
