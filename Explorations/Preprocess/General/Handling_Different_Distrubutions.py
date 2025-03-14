import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
np.random.seed(42)

# Create mock dataset1
num_samples1 = 1000
features1 = np.random.rand(num_samples1, 5)  # 5 features
target1 = np.random.randint(0, 2, num_samples1)  # Binary target

dataset1 = pd.DataFrame(features1, columns=[f'feature_{i}' for i in range(1, 6)])
dataset1['target'] = target1

# Create mock dataset2
num_samples2 = 500
features2 = np.random.rand(num_samples2, 5)  # 5 features
target2 = np.random.randint(0, 2, num_samples2)  # Binary target

dataset2 = pd.DataFrame(features2, columns=[f'feature_{i}' for i in range(1, 6)])
dataset2['target'] = target2

# Combine datasets
combined_data = pd.concat([dataset1, dataset2], ignore_index=True)

# Shuffle and split into train, dev, and test sets
train_data, temp_data = train_test_split(combined_data, test_size=0.3, random_state=42)
dev_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Separate features and target
X_train, y_train = train_data.drop('target', axis=1), train_data['target']
X_dev, y_dev = dev_data.drop('target', axis=1), dev_data['target']
X_test, y_test = test_data.drop('target', axis=1), test_data['target']

# Model tuning using the dev set
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_dev, y_dev)

# Best parameters from tuning
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Train the model using the best parameters on the full training set
best_rf = RandomForestClassifier(**best_params, random_state=42)
best_rf.fit(X_train, y_train)

# Validate the model with the dev set
dev_predictions = best_rf.predict(X_dev)
dev_accuracy = accuracy_score(y_dev, dev_predictions)
print(f"Dev Set Accuracy: {dev_accuracy:.2f}")

# Test the model with the test set
test_predictions = best_rf.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"Test Set Accuracy: {test_accuracy:.2f}")

# Visualization: Plot feature distributions and model performance
plt.figure(figsize=(12, 6))

# Plot feature distributions
plt.subplot(1, 2, 1)
plt.hist([dataset1['feature_1'], dataset2['feature_1']], bins=20, label=['Dataset 1', 'Dataset 2'], alpha=0.7)
plt.title('Feature 1 Distribution')
plt.xlabel('Feature 1 Value')
plt.ylabel('Frequency')
plt.legend()

# Plot model performance
plt.subplot(1, 2, 2)
plt.bar(['Dev Set', 'Test Set'], [dev_accuracy, test_accuracy], color=['blue', 'green'])
plt.title('Model Accuracy')
plt.ylim(0, 1)
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()