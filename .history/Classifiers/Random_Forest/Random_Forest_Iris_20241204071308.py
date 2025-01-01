'''
Use famous iris flower dataset from sklearn.datasets to predict flower 
species using random forest classifier.
1. Measure prediction score using default n_estimators (10)
2. Now fine tune your model by changing number of trees in your classifer and tell me
 what best score you can get using how many trees


 Results change dependnat on the random state, why?
'''
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Define features and target
X = df.drop('target', axis='columns')
y = df.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# Model 1: Default n_estimators = 10
model_1 = RandomForestClassifier(n_estimators=10, random_state=47)
model_1.fit(X_train, y_train)
y_pred_1 = model_1.predict(X_test)

# Cross-validation for default model
cv_scores_1 = cross_val_score(model_1, X, y, cv=5)
print(f"Mean cross-validation accuracy for Model 1: {cv_scores_1.mean():.2f}")

# Fine-Tuning n_estimators with Cross-Validation
n_estimators_range = range(2, 51, 2)
cv_means = []
test_scores = []

for n in n_estimators_range:
    model = RandomForestClassifier(n_estimators=n, random_state=47)
    
    # Cross-validation accuracy
    cv_scores = cross_val_score(model, X, y, cv=5)
    cv_mean = cv_scores.mean()
    cv_means.append(cv_mean)
    
    # Train-test split accuracy
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)
    test_scores.append(test_score)

# Identify best number of trees based on cross-validation
best_cv_score = max(cv_means)
best_n_estimators_cv = n_estimators_range[cv_means.index(best_cv_score)]
print(f"\nBest cross-validation accuracy: {best_cv_score:.2f} achieved with n_estimators={best_n_estimators_cv}")

# Plot accuracy vs n_estimators
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, cv_means, marker='o', label='Cross-Validation Accuracy')
plt.plot(n_estimators_range, test_scores, marker='x', label='Test Set Accuracy')
plt.title("Accuracy vs Number of Trees (n_estimators)")
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()
