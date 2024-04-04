# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import GridSearchCV

# Loading the Titanic dataset
df = pd.read_csv("data/titanic.csv")
df['Sex'] = df['Sex'].map({'male': 1, 'female': 2})
df['Age'].fillna(df['Age'].mean(), inplace=True)

#TASK!! Given: ['Pclass','Age','Fare','Survived']
# Predict the passengers sex

features=['Pclass','Age','Fare','Survived','Sex']
df=df[features]


# Defining inputs and target
X = df.drop('Survived', axis='columns')
y = df['Survived']

# Define the hyperparameters grid
param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10,20],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize Decision Tree Classifier model
model = tree.DecisionTreeClassifier()

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters found
print("\n Best Parameters:")
for k,v in grid_search.best_params_.items():
    print(f"{k}: {v}")

# Get the best estimator
best_model = grid_search.best_estimator_



# Evaluate the best model
best_model.fit(X_train, y_train)
accuracy = best_model.score(X_test, y_test)
print("Best Model Accuracy:", accuracy)
