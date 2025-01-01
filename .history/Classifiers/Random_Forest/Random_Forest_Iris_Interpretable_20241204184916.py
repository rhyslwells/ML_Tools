import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import export_graphviz
import graphviz

# Load the iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Define features and target
X = df.drop('target', axis='columns')
y = df['target']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tune the number of estimators using cross-validation
estimators = range(1, 101, 5)  # Test values from 1 to 100 in steps of 5
cv_scores = []

for n in estimators:
    model = RandomForestClassifier(n_estimators=n, random_state=42)
    scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation
    cv_scores.append(scores.mean())

# Find the best number of estimators
best_score = max(cv_scores)
best_n_estimators = estimators[cv_scores.index(best_score)]
print(f"\nBest Cross-Validation Accuracy: {best_score:.4f} achieved with n_estimators={best_n_estimators}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(estimators, cv_scores, marker='o')
plt.title("Random Forest Cross-Validation Accuracy vs. Number of Estimators")
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Cross-Validation Accuracy")
plt.grid(True)
plt.show()

# Train final model with the best number of estimators
model_best = RandomForestClassifier(n_estimators=best_n_estimators, random_state=42)
model_best.fit(X_train, y_train)
y_pred_best = model_best.predict(X_test)


# Visualize a single decision tree
tree = model_best.estimators_[0]  # First tree in the forest
dot_data = export_graphviz(
    tree,
    out_file=None,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    special_characters=True,
)

# Visualize with Graphviz
graph = graphviz.Source(dot_data)
graph.view()

# End of script


#-------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import export_graphviz
import graphviz
from sklearn.inspection import plot_partial_dependence


# Load the Iris dataset
iris = load_iris()

# Create a DataFrame from the Iris data
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Define features (X) and target (y)
X = df.drop('target', axis='columns')
y = df['target']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_n_estimators=11
# Train the final model with the best number of estimators
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
plot_partial_dependence(model_best, X, features=[0, 1], feature_names=iris.feature_names)
plt.show()

# Visualize a single decision tree from the Random Forest
tree = model_best.estimators_[0]  # Access the first tree in the forest
dot_data = export_graphviz(
    tree,
    out_file=None,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    special_characters=True,
)

# Generate and render the decision tree visualization
graph = graphviz.Source(dot_data)
graph.view()  # This will open the tree visualization in a viewer

# Optional: Visualize multiple decision trees (first 3 trees)
for i, tree in enumerate(model_best.estimators_[:3]):  # Visualize first 3 trees
    dot_data = export_graphviz(
        tree,
        out_file=None,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True,
        rounded=True,
        special_characters=True,
    )
    
    graph = graphviz.Source(dot_data)
    graph.render(f"tree_{i}")  # Saves the tree visualization as a file tree_0, tree_1, etc.

# End of script
