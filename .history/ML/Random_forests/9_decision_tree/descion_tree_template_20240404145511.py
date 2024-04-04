# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree

# Loading the Titanic dataset
df = pd.read_csv("titanic.csv")

# Dropping unnecessary columns
df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns', inplace=True)

# Preprocessing data
df['Sex'] = df['Sex'].map({'male': 1, 'female': 2})
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Defining inputs and target
inputs = df.drop('Survived', axis='columns')
target = df['Survived']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2)

# Initializing Decision Tree Classifier model
model = tree.DecisionTreeClassifier()

# Training the model
model.fit(X_train, y_train)

# Evaluating the model
score = model.score(X_test, y_test)
print("Model Score:", score)

# Making predictions
predictions = model.predict(X_test)

# Further analysis or evaluation can be performed here
