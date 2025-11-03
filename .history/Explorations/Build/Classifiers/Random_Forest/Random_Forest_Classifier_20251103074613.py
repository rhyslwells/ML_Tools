"""
Exploration: Random Forest Classifier on the Iris Dataset
--------------------------------------------------------

Objective:
1. Use the Iris dataset from sklearn.datasets to predict flower species.
2. Measure prediction accuracy using a default number of trees (n_estimators=10).
3. Fine-tune the model by varying the number of trees and determine the best-performing configuration.
4. Explore how results depend on the random_state and cross-validation settings.

Key Questions:
- How does increasing the number of trees affect accuracy?
- Why do results vary depending on the random_state?
"""

#---------------------------------------------
# Imports
#---------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

#---------------------------------------------
# Load and Inspect Dataset
#---------------------------------------------
iris = load_iris()

# Convert to DataFrame for easy handling
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Examine structure
print(df.head())
print("Target classes:", iris.target_names)

#---------------------------------------------
# Define Features (X) and Target (y)
#---------------------------------------------
X = df.drop('target', axis=1)
y = df['target']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#---------------------------------------------
# Model 1: Default Random Forest (n_estimators=10)
#---------------------------------------------
print("\n=== Model 1: Default Random Forest (10 Trees) ===")
model_1 = RandomForestClassifier(n_estimators=10, random_state=47)
model_1.fit(X_train, y_train)
y_pred_1 = model_1.predict(X_test)

# Evaluate classification performance
print(classification_report(y_test, y_pred_1))

# 5-Fold Cross-Validation
cv_scores_1 = cross_val_score(model_1, X, y, cv=5)
print(f"Mean CV Accuracy (Model 1): {cv_scores_1.mean():.2f}")

#---------------------------------------------
# Model Tuning: Vary n_estimators
#---------------------------------------------
print("\n=== Fine-Tuning Number of Trees ===")

n_estimators_range = range(2, 51, 2)
cv_means, test_scores = [], []

for n in n_estimators_range:
    model = RandomForestClassifier(n_estimators=n, random_state=np.random.RandomState(0))
    
    # Cross-validation mean accuracy
    cv_mean = cross_val_score(model, X, y, cv=5).mean()
    cv_means.append(cv_mean)
    
    # Test accuracy after fitting
    model.fit(X_train, y_train)
    test_scores.append(accuracy_score(y_test, model.predict(X_test)))

# Identify best number of trees
best_cv_score = max(cv_means)
best_n_estimators = n_estimators_range[cv_means.index(best_cv_score)]
print(f"Best CV Accuracy: {best_cv_score:.2f} with n_estimators={best_n_estimators}")

# Plot: Accuracy vs Number of Trees
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, cv_means, marker='o', label='Cross-Validation Accuracy')
plt.plot(n_estimators_range, test_scores, marker='x', label='Test Accuracy')
plt.title("Accuracy vs Number of Trees (n_estimators)")
plt.xlabel("Number of Trees")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

#---------------------------------------------
# Exploring Effect of Random State and CV Splits
#---------------------------------------------
print("\n=== Exploring Random State and Cross-Validation Settings ===")

rng_nums = [0]
cv_nums = [5, 10]
results = {}

for rng_num in rng_nums:
    for cv_num in cv_nums:
        cv_means, test_scores = [], []
        
        for n in n_estimators_range:
            rng = np.random.RandomState(rng_num)
            model = RandomForestClassifier(n_estimators=n, random_state=rng)
            
            # Compute accuracies
            cv_mean = cross_val_score(model, X, y, cv=cv_num).mean()
            model.fit(X_train, y_train)
            test_score = accuracy_score(y_test, model.predict(X_test))
            
            cv_means.append(cv_mean)
            test_scores.append(test_score)
        
        results[(rng_num, cv_num)] = {'cv_means': cv_means, 'test_scores': test_scores}

# Plot results for different settings
plt.figure(figsize=(12, 8))
for (rng_num, cv_num), metrics in results.items():
    label = f"RNG={rng_num}, CV={cv_num}"
    plt.plot(n_estimators_range, metrics['cv_means'], marker='o', label=f"{label} (CV Mean)")
    plt.plot(n_estimators_range, metrics['test_scores'], marker='x', linestyle='--', label=f"{label} (Test Score)")

plt.title("Accuracy vs Number of Trees (Varying RNG and CV Settings)")
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

#---------------------------------------------
# Comparing Models with Different n_estimators
#---------------------------------------------
print("\n=== Comparing Classification Reports ===")

# Model with fewer trees
model_1 = RandomForestClassifier(n_estimators=2, random_state=3)
model_1.fit(X_train, y_train)
print("Model 1 (2 Trees):")
print(classification_report(y_test, model_1.predict(X_test)))

# Model with more trees
model_2 = RandomForestClassifier(n_estimators=10, random_state=3)
model_2.fit(X_train, y_train)
print("\nModel 2 (10 Trees):")
print(classification_report(y_test, model_2.predict(X_test)))

#---------------------------------------------
# Checking for Overfitting (Cross-Validation)
#---------------------------------------------
print("\n=== Overfitting Check (Cross-Validation on Model 2) ===")
cv_scores = cross_val_score(model_2, X, y, cv=5)
print("CV Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())

#---------------------------------------------
# Increasing Estimators Further (Model 3)
#---------------------------------------------
print("\n=== Model 3: Increased Number of Trees (20) ===")
model_3 = RandomForestClassifier(n_estimators=20, random_state=3)
model_3.fit(X_train, y_train)

# Evaluate training vs testing accuracy
train_acc = accuracy_score(y_train, model_3.predict(X_train))
test_acc = accuracy_score(y_test, model_3.predict(X_test))

print(f"Training Accuracy: {train_acc:.2f}")
print(f"Test Accuracy: {test_acc:.2f}")

# Cross-validation for Model 3
cv_scores_3 = cross_val_score(model_3, X, y, cv=5)
print("CV Scores:", cv_scores_3)
print(f"Mean CV Score: {cv_scores_3.mean():.2f}")

"""
Summary Notes:
--------------
- Increasing n_estimators generally improves model stability and accuracy until performance plateaus.
- Random state affects results because it controls the randomness in tree sampling and bootstrapping.
- Cross-validation provides a more robust measure of generalization compared to a single test split.
"""
