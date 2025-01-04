'''
Use famous iris flower dataset from sklearn.datasets to predict flower 
species using random forest classifier.
1. Measure prediction score using default n_estimators (10)
2. Now fine tune your model by changing number of trees in your classifer and tell me
 what best score you can get using how many trees
'''

import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import confusion_matrix,classification_report

iris = load_iris()
dir(iris)

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()

# get all columns of iris
iris.keys()

# what is the target column
iris.target_names # labeling of certain plants
iris.target

df['target'] = iris.target
df.head()

# **Train and the model and prediction**
X = df.drop('target',axis='columns')
y = df.target

#modeling
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)

# evaluation:
y_predicted = model.predict(X_test)

#What can we seen from the classification report?
print(classification_report(y_test, y_predicted))

#---------------------------------------------------------------------

# Suppose we want to compare classifcation reports for two separate models on the 
# same test data but with more estimators for the second model.

# model 1
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=3)
model_1 = RandomForestClassifier(n_estimators=2)
model_1.fit(X_train, y_train)
y_predicted_1 = model_1.predict(X_test)
print("Classification Report for Model 1:")
print(classification_report(y_test, y_predicted_1))

# Model 2
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=3)
model_2 = RandomForestClassifier(n_estimators=10)
model_2.fit(X_train, y_train)
y_predicted_2 = model_2.predict(X_test)
print("\nClassification Report for Model 2:")
print(classification_report(y_test, y_predicted_2))

#--------------------------------------------------------------------------------

#Q: Is Model 2 overfit? 

from sklearn.model_selection import cross_val_score

# Perform cross-validation
cv_scores = cross_val_score(model_2, X, y, cv=5)  # Use 5-fold cross-validation

# Print cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())


#--------------------------------------------------------------------------------

#Q: What happens to the cross validation if we increase the number of estimators?

from sklearn.metrics import accuracy_score

# Model 3
model_3 = RandomForestClassifier(n_estimators=20, random_state=3)  # Increase the number of estimators
model_3.fit(X_train, y_train)

# Training set performance
y_train_pred_3 = model_3.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred_3)
print("Training Accuracy for Model 3:", train_accuracy)

# Test set performance
y_test_pred_3 = model_3.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred_3)
print("Test Accuracy for Model 3:", test_accuracy)

# Perform cross-validation for Model 3
cv_scores_3 = cross_val_score(model_3, X, y, cv=5)  # Use 5-fold cross-validation

# Print cross-validation scores for Model 3
print("Cross-Validation Scores for Model 3:", cv_scores_3)
print("Mean CV Score for Model 3:", cv_scores_3.mean())
