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


# visualise groupings:

# Loading data
X_rep, X_new, y_rep, y_new = load_data()

# Leaving only first two features
X_rep = X_rep[:, :2]
X_new = X_new[:, :2]

# Selecting random sample
sample_id = 2
sample = X_new[sample_id]
sample_class = y_new[sample_id]

# Creating KNN object
knn = KNN(k=5, distance_func=euclidean_distance)

# Selecting neighbours
knn_distances = knn._calculate_distances(X_rep, sample)
neighbor_ids = knn._get_k_neighbors(knn_distances)

# Visualisation
plt.figure(figsize=(15, 9))
plt.scatter(X_rep[y_rep == 0][:, 0], X_rep[y_rep == 0][:, 1], s=100, label="setosa")
plt.scatter(X_rep[y_rep == 1][:, 0], X_rep[y_rep == 1][:, 1], s=100, label="versicolor")
plt.scatter(X_rep[y_rep == 2][:, 0], X_rep[y_rep == 2][:, 1], s=100, label="virginica")
plt.scatter([sample[0]], [sample[1]], color="red", s=100, label="query_sample", marker="*")
plt.scatter(X_rep[neighbor_ids, 0], X_rep[neighbor_ids, 1], 
            color="black", s=120, label="neighbours", marker="x")
plt.xlabel("sepal length (cm)")
plt.ylabel("sepal length (cm)")
plt.legend();

# You can use difference distance functions (how do the groupings compare?)

