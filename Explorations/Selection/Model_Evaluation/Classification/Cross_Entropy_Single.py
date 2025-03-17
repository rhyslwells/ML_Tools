import numpy as np
import matplotlib.pyplot as plt

# Function to calculate cross entropy loss
def cross_entropy_loss(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return -np.sum(y_true * np.log(y_pred + 1e-15))  # Adding epsilon for numerical stability

# Example: True labels (one-hot encoded) and predicted probabilities
true_labels = [1, 0, 0]  # Class A
predicted_probs = [0.7, 0.2, 0.1]  # Predicted probabilities

# Calculate cross entropy loss
loss = cross_entropy_loss(true_labels, predicted_probs)

# Visualization of probabilities
classes = ['A', 'B', 'C']

# Plot true labels and predicted probabilities
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# True labels
ax[0].bar(classes, true_labels, color='skyblue', alpha=0.7, label='True Labels')
ax[0].set_title('True Label Distribution')
ax[0].set_ylabel('Probability')
ax[0].set_ylim(0, 1.2)
ax[0].legend()

# Predicted probabilities
ax[1].bar(classes, predicted_probs, color='orange', alpha=0.7, label='Predicted Probabilities')
ax[1].set_title('Predicted Probability Distribution')
ax[1].set_ylabel('Probability')
ax[1].set_ylim(0, 1.2)
ax[1].legend()

plt.tight_layout()
plt.show()

# Display cross entropy loss
print(f"Cross Entropy Loss: {loss:.4f}")

# Additional visualization: Loss changes with predicted probabilities
probabilities = np.linspace(0.01, 1, 100)
losses = [-np.log(p) for p in probabilities]

plt.figure(figsize=(8, 6))
plt.plot(probabilities, losses, label='Cross Entropy Loss', color='red')
plt.title('Cross Entropy Loss vs. Predicted Probability')
plt.xlabel('Predicted Probability for True Class')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()
