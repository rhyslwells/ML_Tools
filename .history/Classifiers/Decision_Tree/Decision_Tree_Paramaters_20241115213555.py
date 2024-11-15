# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, log_loss
)
import numpy as np

# Load Titanic dataset
df = pd.read_csv("data/titanic.csv")
df['Sex'] = df['Sex'].map({'male': 1, 'female': 2})
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Define features and target variable
features = ['Pclass', 'Age', 'Fare', 'Sex', 'SibSp', 'Parch']
X = df[features]
y = df['Survived']

# Function to generate a detailed comparison table

def generate_comparison_table(X, y, param_grid, variable):
    """
    Generates a comparison table for various Decision Tree hyperparameter values.

    Parameters:
        X: pd.DataFrame - Feature dataset.
        y: pd.Series - Target variable.
        param_grid: dict - Hyperparameter grid for GridSearchCV.
        variable: str - The hyperparameter to analyze.

    Returns:
        pd.DataFrame - Table with metrics for each parameter value.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize Decision Tree Classifier and GridSearchCV
    clf = DecisionTreeClassifier()
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='f1', cv=5, verbose=1)
    grid_search.fit(X_train, y_train)

    # Create a DataFrame to store the results
    results = pd.DataFrame(columns=[
        'Parameter Value', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC',
        'Log Loss', 'Confusion Matrix', 'Cross-Val Accuracy', 'Tree Depth', 'Number of Leaves'])

    results_list =[]

    # Loop through each parameter combination
    for params in grid_search.cv_results_['params']:
        clf.set_params(**params)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        
        # ROC-AUC only for binary classification
        roc_auc = None
        if len(np.unique(y)) == 2:
            roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

        # Log Loss
        log_loss_value = log_loss(y_test, clf.predict_proba(X_test))

        # Confusion Matrix
        cm = confusion_matrix(y_test, predictions)

        # Cross-Validation Accuracy
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()

        # Model Complexity Metrics
        tree_depth = clf.get_depth()
        num_leaves = clf.get_n_leaves()

        # Append results to the list (using pd.concat later)
        result_row = pd.DataFrame([{
            'Parameter Value': params[variable],
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'ROC-AUC': roc_auc,
            'Log Loss': log_loss_value,
            'Confusion Matrix': cm.tolist(),
            'Cross-Val Accuracy': cv_mean,
            'Tree Depth': tree_depth,
            'Number of Leaves': num_leaves
        }])

        results_list.append(result_row)

    # Concatenate all results
    return pd.concat(results_list, ignore_index=True)

def describe_evaluation_metrics():
    """
    Prints a description of the evaluation metrics used for model evaluation.
    """
    print("Evaluation Metrics Description:\n")
    descriptions = {
        "Accuracy": "The proportion of correctly classified samples out of the total samples.",
        "Precision": "The proportion of correctly predicted positive observations to the total predicted positives. High precision means low false positives.",
        "Recall": "The proportion of correctly predicted positive observations to the actual positives. High recall means low false negatives.",
        "F1": "The harmonic mean of precision and recall, providing a balance between the two metrics.",
        "ROC-AUC": "The area under the Receiver Operating Characteristic curve, indicating the model's ability to distinguish between classes. Higher is better.",
        "Log Loss": "Measures the distance between the predicted probabilities and the true labels. Lower values are better.",
        "Confusion Matrix": "A matrix showing the counts of true positives, true negatives, false positives, and false negatives.",
        "Cross-Val Accuracy": "The mean accuracy over multiple training-validation splits, providing a more robust estimate of model performance.",
        "Tree Depth": "The depth of the decision tree, reflecting its complexity.",
        "Number of Leaves": "The total number of terminal nodes (leaves) in the decision tree, indicating model complexity."
    }

    for metric, description in descriptions.items():
        print(f"- {metric}: {description}")
    print("\n")

# Example 1: Criterion Comparison
param_grid_criterion = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}
comparison_criterion = generate_comparison_table(X, y, param_grid_criterion, 'criterion')
pd.set_option('display.max_rows', None)
describe_evaluation_metrics()
print("\nCriterion Comparison:\n")
pd.set_option('display.max_columns', None)  # Show all columns
comparison_criterion.head()

# Example 2: Max Depth Variation
param_grid_depth = {
    'criterion': ['gini'],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}
comparison_depth = generate_comparison_table(X, y, param_grid_depth, 'max_depth')
describe_evaluation_metrics()
print("\nMax Depth Variation:\n")
comparison_depth.head()

# Example 3: Min Samples Leaf Variation
param_grid_leaf = {
    'criterion': ['gini'],
    'max_depth': [5],
    'min_samples_split': [2],
    'min_samples_leaf': [1, 5, 10]
}
comparison_leaf = generate_comparison_table(X, y, param_grid_leaf, 'min_samples_leaf')
describe_evaluation_metrics()
print("\nMin Samples Leaf Variation:\n")
comparison_leaf.head()
