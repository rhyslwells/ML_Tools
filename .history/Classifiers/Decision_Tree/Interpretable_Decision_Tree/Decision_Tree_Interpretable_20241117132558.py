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
from sklearn.model_selection import cross_val_score


# Load Titanic dataset
df = pd.read_csv("../data/titanic.csv")
# Convert categorical 'Sex' column into numeric values for the model
# One-hot encode 'Sex' and drop redundant columns
df = pd.get_dummies(df, columns=['Sex'], drop_first=True)

# Fill missing values in 'Age' with the mean of the column to handle missing data
df['Age'].fillna(df['Age'].mean(), inplace=True)

df.head()


# Define the features (independent variables) and target (dependent variable)
features = ['Pclass', 'Age', 'Fare', 'Sex_male', 'SibSp', 'Parch']
X = df[features]  # Features
y = df['Survived']  # Target variable (Survived: 1 = Yes, 0 = No)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Grid Search for Hyperparameter Tuning
# Define the parameter grid to search over
param_grid = {
    'max_depth': [2,3, 5, 7, 10, None],  # Depth of the tree (controls model complexity)
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
def evaluate_model_cv(model, X, y, scoring='accuracy', cv=5):
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    for metric in metrics:
        scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
        print(f"Mean {metric}: {scores.mean():.4f} (Std: {scores.std():.4f})")

# Evaluate the best model with cross-validation
evaluate_model_cv(best_model, X, y)

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
    'max_depth': 2,  # Limiting tree depth to 2 for better interpretability (simpler model)
    'min_samples_split': 10,  # Limiting splits to at least 10 samples per node
    'min_samples_leaf': 4,  # Ensuring at least 4 samples in each leaf node to prevent overfitting
    'random_state': 42,  # Setting a random seed for reproducibility
    'criterion': 'gini'  # Using Gini impurity to measure split quality
}
interpretable_model = DecisionTreeClassifier(**base_params)

# Training the simplified model on the training data
interpretable_model.fit(X_train, y_train)
evaluate_model_cv(interpretable_model, X,y)

# Visualizing the simplified model
visualize_tree(interpretable_model, X.columns)

# Question: When building a Decision tree, why does it split based on certain features first? Is this due to the gini value?
# Answer: When building a Decision tree, it splits based on certain features first because the gini value is used to measure the quality of a split. 
# The gini value is a measure of the impurity or uncertainty of a node, which is higher for a node with more classes (e.g., more than two values) than for a node with fewer classes.

# Base-Pruned Interpretable Decision Tree :-----------------------------------------------
# Apply Cost Complexity Pruning (ccp_alpha) to the simplified model for pruning
ccp_alphas = interpretable_model.cost_complexity_pruning_path(X_train, y_train).ccp_alphas
best_alpha = ccp_alphas[-3]  # Choose the largest alpha (prune most branches)
# -1,-2 : remove too many branches. Where -4 returns same tree.
# print("ccp_alpha:", ccp_alphas)

# Refit the model with pruning
pruned_params = base_params.copy()
pruned_params.update({'ccp_alpha': best_alpha})
pruned_interpretable_model = DecisionTreeClassifier(**pruned_params)
pruned_interpretable_model.fit(X_train, y_train)

# Evaluate the pruned model
evaluate_model_cv(pruned_interpretable_model, X,y)

# Visualizing the pruned model
visualize_tree(pruned_interpretable_model, X.columns)

# Base-Pruned-FS Interpretable Decision Tree :-----------------------------------------------
# Feature Selection using Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE

# Create an RFE model and select top 3 features based on the pruned model
rfe = RFE(estimator=pruned_interpretable_model, n_features_to_select=3)
X_rfe = rfe.fit_transform(X, y)

# Split the data into training and testing sets (80% training, 20% testing)
X_train_rfe, X_test_rfe, y_train_rfe, y_test_rfe = train_test_split(X, y, test_size=0.2, random_state=42)


# Train model with selected features
rfe_pruned_interpretable_model = DecisionTreeClassifier(**pruned_params)
rfe_pruned_interpretable_model.fit(X_train_rfe, y_train_rfe)

# Predict and evaluate model with selected features
X_test_rfe = rfe.transform(X_test)
y_pred_rfe = rfe_pruned_interpretable_model.predict(X_test_rfe, y_test)


# Evaluate the pruned model with selected features
evaluate_model_cv(rfe_pruned_interpretable_model, X_rfe,y)

# Visualizing the RFE model
visualize_tree(rfe_pruned_interpretable_model, X.columns)

# Extract Decision Rules
decision_rules = export_text(pruned_interpretable_model, feature_names=features)
print("Decision Rules from Pruned Model:")
print(decision_rules)


