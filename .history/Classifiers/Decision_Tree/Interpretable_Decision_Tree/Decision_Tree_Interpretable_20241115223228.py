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
df = pd.read_csv("../data/titanic.csv")
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

# Documenting What We're Trying to Accomplish
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
# Best Parameters from Grid Search: {'criterion': 'gini', 'max_depth': 3, 'min_samples_leaf': 2, 'min_samples_split': 2}

# 2. Evaluating the Best Model from Grid Search
# Get the best model from grid search
best_model = grid_search.best_estimator_


def evaluate_model(model, X, y):
    # Calculate and print classification metrics for the best model
    # Predict on the test data

    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    log_loss_value = log_loss(y, model.predict_proba(X))

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Log Loss: {log_loss_value:.4f}")

# Print the evaluation metrics for the best model
evaluate_model(best_model, X_test, y_test)
# Accuracy: 0.7989
# Precision: 0.7969
# Recall: 0.6892
# F1 Score: 0.7391
# ROC AUC: 0.8463
# Log Loss: 0.4675


def visualize_tree(model, features):    
    # 5. Visualizing the Decision Trees
    # Visualizing the best model from the grid search
    plt.figure(figsize=(12, 8))
    plt.title("Best Decision Tree from Grid Search")
    plot_tree(best_model, filled=True, feature_names=features, class_names=["Not Survived", "Survived"], rounded=True)
    plt.show()

visualize_tree(best_model, X.columns)

# Base Interpretable Decision Tree :-----------------------------------------------
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

evaluate_model(interpretable_model, X_test, y_test)
# Interpretable Model - Accuracy: 0.7989
# Interpretable Model - Precision: 0.7969
# Interpretable Model - Recall: 0.6892
# Interpretable Model - F1 Score: 0.7391
# Interpretable Model - ROC AUC: 0.8456
# Interpretable Model - Log Loss: 0.4787

visualize_tree(interpretable_model, X.columns)


# Base-Pruned Interpretable Decision Tree :-----------------------------------------------
# Now, apply Cost Complexity Pruning (ccp_alpha) to the simplified model for pruning
ccp_alphas = interpretable_model.cost_complexity_pruning_path(X_train, y_train).ccp_alphas
best_alpha = ccp_alphas[-1]  # Choose the largest alpha (prune most branches)

# Refit the model with pruning
pruned_interpretable_model = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
pruned_interpretable_model.fit(X_train, y_train)

# Evaluate the pruned model
evaluate_model(pruned_interpretable_model, X_test, y_test)
visualize_tree(pruned_interpretable_model, X.columns)


# Base-Pruned-FS Interpretable Decision Tree :-----------------------------------------------
# Feature Selection using Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE

# Create an RFE model and select top 3 features based on the pruned model
rfe = RFE(estimator=pruned_interpretable_model, n_features_to_select=3)
X_rfe = rfe.fit_transform(X_train, y_train)

# Train model with selected features
rfe_pruned_interpretable_model = DecisionTreeClassifier(random_state=42)
rfe_pruned_interpretable_model.fit(X_rfe, y_train)

# Predict and evaluate model with selected features
X_test_rfe = rfe.transform(X_test)
y_pred_rfe = rfe_pruned_interpretable_model.predict(X_test_rfe)
accuracy_rfe = accuracy_score(y_test, y_pred_rfe)
print(f"RFE Model - Accuracy with selected features: {accuracy_rfe:.4f}")

# Evaluate the pruned model with selected features
evaluate_model(rfe_pruned_interpretable_model, X_test_rfe, y_test)
visualize_tree(rfe_pruned_interpretable_model, X.columns)



# Later


# 7. Extract Decision Rules
# Using export_text to extract decision rules from the pruned model
decision_rules = export_text(pruned_model, feature_names=features)
print("Decision Rules from Pruned Model:")
print(decision_rules)

