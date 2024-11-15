# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, log_loss
)
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# Load Titanic dataset
df = pd.read_csv("data/titanic.csv")
# Convert categorical 'Sex' column into numeric values for the model
df['Sex'] = df['Sex'].map({'male': 1, 'female': 2})
# Fill missing values in 'Age' with the mean of the column to handle missing data
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Define the features (independent variables) and target (dependent variable)
features = ['Pclass', 'Age', 'Fare', 'Sex', 'SibSp', 'Parch']
X = df[features]  # Features
y = df['Survived']  # Target variable (Survived: 1 = Yes, 0 = No)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Grid Search for Hyperparameter Tuning
# Define the parameter grid to search over
param_grid = {
    'max_depth': [3, 5, 7, 10, None],  # Depth of the tree (controls model complexity)
    'min_samples_split': [2, 5, 10],  # Minimum samples to split a node (controls overfitting)
    'min_samples_leaf': [1, 2, 4],    # Minimum samples in a leaf node (controls model complexity)
    'criterion': ['gini', 'entropy']  # The function to measure the quality of a split
}

# Setting up GridSearchCV with DecisionTreeClassifier
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, scoring='accuracy')

# Fitting the grid search to the training data
grid_search.fit(X_train, y_train)

# Output the best parameters found by the grid search
print("Best Parameters from Grid Search:", grid_search.best_params_)

# 2. Evaluating the Best Model from Grid Search
# Get the best model from grid search
best_model = grid_search.best_estimator_

# Predict on the test data
y_pred = best_model.predict(X_test)

# Calculate and print classification metrics for the best model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
log_loss_value = log_loss(y_test, best_model.predict_proba(X_test))

# Print the evaluation metrics for the best model
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Log Loss: {log_loss_value:.4f}")

# 3. Making the Model More Interpretable
# Now, let's simplify the model to make it more interpretable, while trying to retain performance
# We will limit the tree depth, the minimum samples required to split a node, and the leaf size
interpretable_model = DecisionTreeClassifier(
    max_depth=3,  # Limiting tree depth to 3 for better interpretability (simpler model)
    min_samples_split=10,  # Limiting splits to at least 10 samples per node
    min_samples_leaf=4,  # Ensuring at least 4 samples in each leaf node to prevent overfitting
    random_state=42,  # Setting a random seed for reproducibility
    criterion='gini'  # Using Gini impurity to measure split quality
)

# Training the simplified model on the training data
interpretable_model.fit(X_train, y_train)

# 4. Evaluating the Simplified Model
# Predict on the test data using the simplified model
y_pred_interpretable = interpretable_model.predict(X_test)

# Calculate and print classification metrics for the simplified model
accuracy_interpretable = accuracy_score(y_test, y_pred_interpretable)
precision_interpretable = precision_score(y_test, y_pred_interpretable)
recall_interpretable = recall_score(y_test, y_pred_interpretable)
f1_interpretable = f1_score(y_test, y_pred_interpretable)
roc_auc_interpretable = roc_auc_score(y_test, interpretable_model.predict_proba(X_test)[:, 1])
log_loss_interpretable = log_loss(y_test, interpretable_model.predict_proba(X_test))

# Print the evaluation metrics for the simplified model
print(f"Interpretable Model - Accuracy: {accuracy_interpretable:.4f}")
print(f"Interpretable Model - Precision: {precision_interpretable:.4f}")
print(f"Interpretable Model - Recall: {recall_interpretable:.4f}")
print(f"Interpretable Model - F1 Score: {f1_interpretable:.4f}")
print(f"Interpretable Model - ROC AUC: {roc_auc_interpretable:.4f}")
print(f"Interpretable Model - Log Loss: {log_loss_interpretable:.4f}")

# 5. Visualizing the Decision Trees
# Visualizing the best model from the grid search
plt.figure(figsize=(12, 8))
plt.title("Best Decision Tree from Grid Search")
plot_tree(best_model, filled=True, feature_names=features, class_names=["Not Survived", "Survived"], rounded=True)
plt.show()

# Visualizing the simplified, more interpretable model
plt.figure(figsize=(12, 8))
plt.title("Interpretable Decision Tree")
plot_tree(interpretable_model, filled=True, feature_names=features, class_names=["Not Survived", "Survived"], rounded=True)
plt.show()

# 6. Documenting What We're Trying to Accomplish
"""
We are performing the following steps:
1. **Hyperparameter Tuning**: Using GridSearchCV to find the optimal hyperparameters for a Decision Tree model.
   - This step ensures that we are using the best configuration for the model.
2. **Evaluating Model Performance**: After training the best model, we evaluate it using standard classification metrics like accuracy, precision, recall, F1-score, ROC AUC, and log loss.
3. **Simplifying the Model**: To make the model more interpretable, we reduce the tree depth and control the size of the leaves and the number of splits.
4. **Evaluating the Simplified Model**: We evaluate the performance of the simplified model to ensure it still performs well in comparison to the original model.
5. **Visualizing the Trees**: We visualize both the complex (best from GridSearch) and simplified decision trees to better understand the decision-making process.
6. **Objective**: The goal is to strike a balance between model interpretability and maintaining good performance on classification metrics. A simpler model should ideally be just as effective but easier to interpret.
"""

# Post-Pruning using Cost Complexity Pruning (ccp_alpha)
# To prune the tree, use cost complexity pruning to remove branches that have little predictive power
ccp_alphas = best_model.cost_complexity_pruning_path(X_train, y_train).ccp_alphas
best_alpha = ccp_alphas[-1]  # Choose the largest alpha (prune most branches)

# Refit the model with pruning
pruned_model = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
pruned_model.fit(X_train, y_train)

# Evaluate pruned model
y_pred_pruned = pruned_model.predict(X_test)
accuracy_pruned = accuracy_score(y_test, y_pred_pruned)
print(f"Pruned Model - Accuracy: {accuracy_pruned:.4f}")

# Feature Selection using Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE

# Create an RFE model and select top 3 features
rfe = RFE(estimator=DecisionTreeClassifier(random_state=42), n_features_to_select=3)
X_rfe = rfe.fit_transform(X_train, y_train)

# Train model with selected features
rfe_model = DecisionTreeClassifier(random_state=42)
rfe_model.fit(X_rfe, y_train)

# Predict and evaluate model with selected features
X_test_rfe = rfe.transform(X_test)
y_pred_rfe = rfe_model.predict(X_test_rfe)
accuracy_rfe = accuracy_score(y_test, y_pred_rfe)
print(f"RFE Model - Accuracy with selected features: {accuracy_rfe:.4f}")

# 7. Extract Decision Rules
# Using export_text to extract decision rules from the pruned model
decision_rules = export_text(pruned_model, feature_names=features)
print("Decision Rules from Pruned Model:")
print(decision_rules)

