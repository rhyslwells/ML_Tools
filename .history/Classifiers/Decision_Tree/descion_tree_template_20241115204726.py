# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score



# Loading the Titanic dataset
df = pd.read_csv("data/titanic.csv")
df.head()
df['Sex'] = df['Sex'].map({'male': 1, 'female': 2})
df['Age'].fillna(df['Age'].mean(), inplace=True)


#Pclass,Sex,Age,SibSp,Parch,Fare,Cabin
features = ['Pclass', 'Age', 'Fare', 'Sex', 'SibSp', 'Parch']  # Updated features
X=df[features]
y=df['Survived']

# Function to generate a detailed comparison table
def generate_comparison_table(X, y, param_grid,variable):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize Decision Tree Classifier
    clf = DecisionTreeClassifier()

    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='f1', cv=5, verbose=1)
    grid_search.fit(X_train, y_train)

    # Create a DataFrame to store the results
    results = pd.DataFrame(columns=['Parameters', 'Accuracy', 'Precision', 'Recall', 'F1'])

    # Loop through all parameter combinations tested
    for params, mean_score, std_score in zip(
        grid_search.cv_results_['params'],
        grid_search.cv_results_['mean_test_score'],
        grid_search.cv_results_['std_test_score'],
    ):
        # Train and evaluate model with these parameters
        clf.set_params(**params)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        # Append to results
        results = results.append({
            'Parameters': params[variable],
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }, ignore_index=True)

    return results

# Example parameter grid
param_grid_criterion = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}

# Generate and display the comparison table
comparison_table = generate_comparison_table(X, y, param_grid_criterion,'criterion')
print(comparison_table)

#show full table
pd.set_option('display.max_rows', None)
print(comparison_table)


#------------------------------------------------------

# Depth variation
param_grid_depth = {
    'criterion': ['gini'],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}

# Leaf size variation
param_grid_leaf = {
    'criterion': ['gini'],
    'max_depth': [5],
    'min_samples_split': [2],
    'min_samples_leaf': [1, 5, 10]
}

print("\nMax Depth Variation:")
Generate_model_with_params(X, y, param_grid_depth)

print("\nMin Samples Leaf Variation:")
Generate_model_with_params(X, y, param_grid_leaf)
