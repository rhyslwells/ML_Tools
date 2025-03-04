# Import required libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])

# Split data into features and target
X = df.drop(['target', 'flower_name'], axis=1)
y = df.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train and evaluate SVM with different parameters
def explore_svm(X_train, X_test, y_train, y_test):
    results = []
    
    # Test different kernels
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    for kernel in kernels:
        model = SVC(kernel=kernel)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append((f"Kernel: {kernel}", acc))
    
    # Test different regularization (C) values
    for C in [0.1, 1, 10, 100]:
        model = SVC(C=C, kernel='rbf')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append((f"Kernel: rbf, C: {C}", acc))
    
    # Test different gamma values
    for gamma in [0.01, 0.1, 1, 10]:
        model = SVC(kernel='rbf', gamma=gamma)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append((f"Kernel: rbf, Gamma: {gamma}", acc))
    
    return results

# Run the exploration
results = explore_svm(X_train, X_test, y_train, y_test)

# Display results
print("Exploration Results:")
for desc, acc in results:
    print(f"{desc} - Accuracy: {acc:.2f}")

# Visualize the best SVM
best_model = SVC(kernel='rbf', C=1, gamma=0.1)
best_model.fit(X_train, y_train)

# Visualize predictions on a 2D feature subset (for simplicity)
X_train_2D = X_train.iloc[:, :2]
X_test_2D = X_test.iloc[:, :2]
best_model.fit(X_train_2D, y_train)

# Create a grid for decision boundary visualization
x_min, x_max = X_train_2D.iloc[:, 0].min() - 1, X_train_2D.iloc[:, 0].max() + 1
y_min, y_max = X_train_2D.iloc[:, 1].min() - 1, X_train_2D.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = best_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary and data points
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X_train_2D.iloc[:, 0], X_train_2D.iloc[:, 1], c=y_train, edgecolor='k', cmap=plt.cm.coolwarm)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('SVM Decision Boundary')
plt.show()
