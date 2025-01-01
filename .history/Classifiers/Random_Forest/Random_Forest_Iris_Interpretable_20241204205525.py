import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.inspection import PartialDependenceDisplay

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Define features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
best_n_estimators = 11
model_best = RandomForestClassifier(n_estimators=best_n_estimators, random_state=42)
model_best.fit(X_train, y_train)

# Visualizing Feature Importance
importances = model_best.feature_importances_
indices = np.argsort(importances)[::-1]

# Plotting feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.barh(range(X.shape[1]), importances[indices], align="center")
plt.yticks(range(X.shape[1]), [iris.feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

# Visualizing Partial Dependence for the top 2 features
# Specify the target class, e.g., the first class (0) in the Iris dataset
PartialDependenceDisplay.from_estimator(
    model_best, 
    X, 
    features=[2,3], 
    feature_names=iris.feature_names,
    target=0  # Specify the target class (0 is the first class)
)
plt.show()
