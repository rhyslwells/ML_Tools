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

# what is the target column
iris.target_names # labeling of certain plants


df['target'] = iris.target
df.head()

# **Train and the model and prediction**
X = df.drop('target',axis='columns')
y = df.target

#modeling
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
model = RandomForestClassifier(n_estimators=20)
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
