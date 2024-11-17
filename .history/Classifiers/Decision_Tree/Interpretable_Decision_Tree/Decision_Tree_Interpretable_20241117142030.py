# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, log_loss
)
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
import numpy as np

# Load Titanic dataset
df = pd.read_csv("../data/titanic.csv")
# Convert categorical 'Sex' column into numeric values for the model
df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
# Fill missing values in 'Age' with the mean of the column to handle missing data
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Define the features (independent variables) and target (dependent variable)
features = ['Pclass', 'Age', 'Fare', 'Sex_male', 'SibSp', 'Parch']
X = df[features]  # Features
y = df['Survived']  # Target variable (Survived: 1 = Yes, 0 = No)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 1: Grid Search for Hyperparameter Tuning
param_grid = {
    'max_depth': [2, 3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Extract the best model and parameters
best_model = grid_search.best_estimator_
best_params = best_model.get_params()
print("Best Parameters from Grid Search:", best_params)

# Evaluation function
def evaluate_model_cv(model, X, y, scoring='accuracy', cv=5):
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    for metric in metrics:
        scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
        print(f"Mean {metric}: {scores.mean():.4f} (Std: {scores.std():.4f})")

# Visualise tree

def visualize_tree(model, features):    
    plt.figure(figsize=(12, 8))
    plot_tree(model, filled=True, feature_names=features, class_names=["Not Survived", "Survived"], rounded=True)
    plt.show()

# Step 2: Feature Selection using RFE
# Step 2: Feature Selection using RFE

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Initialize variables
num_features = range(1, len(features) + 1)
performance_scores = []

# Iterate over different numbers of features
for n in num_features:
    # Apply Recursive Feature Elimination (RFE) with 'n' features
    rfe = RFE(estimator=best_model, n_features_to_select=n)
    rfe.fit(X_train, y_train)
    
    # Transform the training and testing datasets
    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test)
    
    # Train and evaluate a model with the selected features
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train_rfe, y_train)
    y_pred = model.predict(X_test_rfe)
    
    # Compute and store the performance score (e.g., accuracy)
    performance_scores.append(accuracy_score(y_test, y_pred))

# Determine the optimal number of features
optimal_num_features = num_features[np.argmax(performance_scores)]
print(f"Optimal number of features: {optimal_num_features}")

# Plot performance scores vs. the number of features
plt.figure(figsize=(8, 6))
plt.plot(num_features, performance_scores, marker='o', linestyle='-', color='b')
plt.xlabel("Number of Features", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title("Model Performance vs. Number of Features", fontsize=14)
plt.grid(True)
plt.show()

# Apply RFE with the optimal number of features
rfe = RFE(estimator=best_model, n_features_to_select=optimal_num_features)
rfe.fit(X_train, y_train)

# Extract and display the selected features
selected_features = [features[i] for i in range(len(features)) if rfe.support_[i]]
print("Selected Features from RFE:", selected_features)

# Transform datasets using selected features
X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)

# Step 3: Train Base Interpretable Model with Selected Features
base_params = {
    'max_depth': 3,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'random_state': 42,
    'criterion': 'gini'
}
base_interpretable_model = DecisionTreeClassifier(**base_params)
base_interpretable_model.fit(X_train_rfe, y_train)

# Evaluate the base model
print("Base Model Performance with Selected Features:")
evaluate_model_cv(base_interpretable_model, X_train_rfe, y_train)
visualize_tree(base_interpretable_model, selected_features)

# Step 4: Prune the Base Model and Select the Best `ccp_alpha`
ccp_path = base_interpretable_model.cost_complexity_pruning_path(X_train_rfe, y_train)
ccp_alphas = ccp_path.ccp_alphas

# Store results for each `ccp_alpha`
alpha_scores = []

# Evaluate models for each `ccp_alpha`
for alpha in ccp_alphas:
    pruned_params = base_params.copy()
    pruned_params.update({'ccp_alpha': alpha})
    pruned_model = DecisionTreeClassifier(**pruned_params)
    scores = cross_val_score(pruned_model, X_train_rfe, y_train, cv=5, scoring='accuracy')  # Use your preferred metric
    alpha_scores.append((alpha, scores.mean(), scores.std()))

# Find the `ccp_alpha` with the highest mean score
alpha_scores = pd.DataFrame(alpha_scores, columns=['ccp_alpha', 'mean_score', 'std_dev'])
best_alpha_row = alpha_scores.loc[alpha_scores['mean_score'].idxmax()]
best_alpha = best_alpha_row['ccp_alpha']

print(f"Best ccp_alpha: {best_alpha} with mean accuracy: {best_alpha_row['mean_score']:.4f}")

# Refit the pruned model with the best `ccp_alpha`
pruned_params.update({'ccp_alpha': best_alpha})
pruned_interpretable_model = DecisionTreeClassifier(**pruned_params)
pruned_interpretable_model.fit(X_train_rfe, y_train)

# Evaluate the pruned model
print("Pruned Base Model Performance with Selected Features and Best ccp_alpha:")
evaluate_model_cv(pruned_interpretable_model, X_train_rfe, y_train)

# Step 5: Extract Decision Rules from Pruned Model
decision_rules = export_text(pruned_interpretable_model, feature_names=selected_features)
print("Decision Rules from Pruned Model:")
print(decision_rules)
