import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load Iris dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])


# Visualize Sepal Length vs Sepal Width for Setosa vs Versicolor
#The first 50 are setosa and the next 50 are versicolor and the last 50 are virginica

plt.figure(figsize=(8, 6))
plt.scatter(df[:50]['sepal length (cm)'], df[:50]['sepal width (cm)'], color="green", marker='+', label='Setosa')
plt.scatter(df[50:100]['sepal length (cm)'], df[50:100]['sepal width (cm)'], color="blue", marker='.', label='Versicolor')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()

# Split data for training and testing
X = df.drop(['target', 'flower_name'], axis=1)
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Build and train KNN model
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Produce confusion matrix
cm = confusion_matrix(y_test, y_pred)
#explain output


plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
#explain output



# Visualize prediction groupings
plt.figure(figsize=(15, 9))
plt.scatter(X[y == 0]['sepal length (cm)'], X[y == 0]['sepal width (cm)'], s=100, label="Setosa")
plt.scatter(X[y == 1]['sepal length (cm)'], X[y == 1]['sepal width (cm)'], s=100, label="Versicolor")
plt.scatter(X[y == 2]['sepal length (cm)'], X[y == 2]['sepal width (cm)'], s=100, label="Virginica")
plt.scatter(X_test[y_pred == 0]['sepal length (cm)'], X_test[y_pred == 0]['sepal width (cm)'], color='red', label="Setosa Test")
plt.scatter(X_test[y_pred == 1]['sepal length (cm)'], X_test[y_pred == 1]['sepal width (cm)'], color='orange', label="Versicolor Test")
plt.scatter(X_test[y_pred == 2]['sepal length (cm)'], X_test[y_pred == 2]['sepal width (cm)'], color='purple', label="Virginica Test")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.legend()
plt.show()

# vary k.


# You can use difference distance functions (how do the groupings compare?)
# Compare predictions with different metrics

# knn = KNN(k=5, distance_func=manhattan_distance)
