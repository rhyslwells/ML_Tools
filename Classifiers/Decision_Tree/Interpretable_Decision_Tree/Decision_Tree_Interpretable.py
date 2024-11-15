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

# Evaluation function
def evaluate_model(model, X, y):
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

# Evaluate the best model
evaluate_model(best_model, X_test, y_test)

# 3. Visualizing the Decision Trees
def visualize_tree(model, features):    
    plt.figure(figsize=(12, 8))
    plt.title("Best Decision Tree from Grid Search")
    plot_tree(model, filled=True, feature_names=features, class_names=["Not Survived", "Survived"], rounded=True)
    plt.show()

visualize_tree(best_model, X.columns)

# Base Interpretable Decision Tree :-----------------------------------------------
# Simplifying the model for interpretability
base_params = {
    'max_depth': 3,  # Limiting tree depth to 3 for better interpretability (simpler model)
    'min_samples_split': 10,  # Limiting splits to at least 10 samples per node
    'min_samples_leaf': 4,  # Ensuring at least 4 samples in each leaf node to prevent overfitting
    'random_state': 42,  # Setting a random seed for reproducibility
    'criterion': 'gini'  # Using Gini impurity to measure split quality
}
interpretable_model = DecisionTreeClassifier(**base_params)

# Training the simplified model on the training data
interpretable_model.fit(X_train, y_train)
evaluate_model(interpretable_model, X_test, y_test)

# Visualizing the simplified model
visualize_tree(interpretable_model, X.columns)

# Base-Pruned Interpretable Decision Tree :-----------------------------------------------
# Apply Cost Complexity Pruning (ccp_alpha) to the simplified model for pruning
ccp_alphas = interpretable_model.cost_complexity_pruning_path(X_train, y_train).ccp_alphas
best_alpha = ccp_alphas[-1]  # Choose the largest alpha (prune most branches)

# Refit the model with pruning
pruned_params = base_params.copy()
pruned_params.update({'ccp_alpha': best_alpha})
pruned_interpretable_model = DecisionTreeClassifier(**pruned_params)
pruned_interpretable_model.fit(X_train, y_train)

# Evaluate the pruned model
evaluate_model(pruned_interpretable_model, X_test, y_test)

# Visualizing the pruned model
visualize_tree(pruned_interpretable_model, X.columns)

# Base-Pruned-FS Interpretable Decision Tree :-----------------------------------------------
# Feature Selection using Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE

# Create an RFE model and select top 3 features based on the pruned model
rfe = RFE(estimator=pruned_interpretable_model, n_features_to_select=3)
X_rfe = rfe.fit_transform(X_train, y_train)

# Train model with selected features
rfe_pruned_interpretable_model = DecisionTreeClassifier(**pruned_params)
rfe_pruned_interpretable_model.fit(X_rfe, y_train)

# Predict and evaluate model with selected features
X_test_rfe = rfe.transform(X_test)
y_pred_rfe = rfe_pruned_interpretable_model.predict(X_test_rfe)


# Evaluate the pruned model with selected features
evaluate_model(rfe_pruned_interpretable_model, X_test_rfe, y_test)

# Visualizing the RFE model
visualize_tree(rfe_pruned_interpretable_model, X.columns)

# Extract Decision Rules
decision_rules = export_text(pruned_interpretable_model, feature_names=features)
print("Decision Rules from Pruned Model:")
print(decision_rules)


