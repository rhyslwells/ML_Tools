# From sklearn.datasets load digits dataset and do following
#  1. Classify digits (0 to 9) using KNN classifier. You can use different values for k neighbors and need to figure out a value of K that gives you a maximum score. You can manually try different values of K or use gridsearchcv 
#  1. Plot confusion matrix
#  1. Plot classification report

# [Solution link](https://github.com/codebasics/py/blob/master/ML/17_knn_classification/Exercise/knn_exercise_digits_solution.ipynb)


import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()

df = pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()

df['target'] = iris.target
df.head()

df['flower_name'] =df.target.apply(lambda x: iris.target_names[x])
df.head()

import matplotlib.pyplot as plt

df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]

# **Sepal length vs Sepal Width (Setosa vs Versicolor)**
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color="green",marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],color="blue",marker='.')

from sklearn.model_selection import train_test_split
X = df.drop(['target','flower_name'], axis='columns')
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


#knn

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(X_train, y_train)
knn.score(X_test, y_test)

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(7,5))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

# You can use difference distance functions.

