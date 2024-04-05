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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Build and train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
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

# Vary k
k_values = [6, 10]  # Example list of k values to try
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(f"Classification Report for k={k}:")
    print(classification_report(y_test, y_pred))
#why are the results the same?

# You can use difference distance functions (how do the groupings compare?)
# Different distance functions Compare predictions with different metrics
# Example with Euclidean and Manhattan distances
from sklearn.neighbors import DistanceMetric

distances = ['euclidean', 'manhattan']  # Example list of distance metrics to try

for distance in distances:
    knn = KNeighborsClassifier(n_neighbors=5, metric=distance)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    print(f"Classification Report with {distance.capitalize()} distance:")
    print(classification_report(y_test, y_pred))


# Bad vs Good models --------------------------------------------
from sklearn.model_selection import cross_val_score

# Split data for training and testing
X = df.drop(['target', 'flower_name'], axis=1)
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Bad Model: --------------------------------------------

# Bad Model: n_neighbors=1, weights='uniform'
knn_bad = KNeighborsClassifier(n_neighbors=1, weights='uniform')
knn_bad.fit(X_train, y_train)
y_pred_bad = knn_bad.predict(X_test)

# Evaluate Bad Model with Cross-Validation
scores_bad = cross_val_score(knn_bad, X, y, cv=5, scoring='accuracy')
print("Cross-Validation Scores for Bad Model (n_neighbors=1, weights='uniform'):", scores_bad)
print("Mean Accuracy:", scores_bad.mean())

# Evaluate Bad Model
print("Classification Report for Bad Model (n_neighbors=1, weights='uniform'):")
print(classification_report(y_test, y_pred_bad))
print("Confusion Matrix for Bad Model:")
cm_bad = confusion_matrix(y_test, y_pred_bad)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_bad, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Confusion Matrix for Bad Model')
plt.show()

#Good Model: --------------------------------------------

# Good Model: n_neighbors=15, weights='distance'
knn_good = KNeighborsClassifier(n_neighbors=15, weights='distance')
knn_good.fit(X_train, y_train)
y_pred_good = knn_good.predict(X_test)

# Evaluate Good Model with Cross-Validation
scores_good = cross_val_score(knn_good, X, y, cv=5, scoring='accuracy')
print("Cross-Validation Scores for Good Model (n_neighbors=15, weights='distance'):", scores_good)
print("Mean Accuracy:", scores_good.mean())


# Evaluate Good Model
print("Classification Report for Good Model (n_neighbors=15, weights='distance'):")
print(classification_report(y_test, y_pred_good))
print("Confusion Matrix for Good Model:")
cm_good = confusion_matrix(y_test, y_pred_good)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_good, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Confusion Matrix for Good Model')
plt.show()

#notes


# Bad Model:

# 1. **Overfitting**: With `n_neighbors=1`, the model considers only the nearest neighbor when making predictions.
#  This hyperparameter choice causes the decision boundary to be highly sensitive to individual data points,
# leading to overfitting.

# 2. **Vulnerability to Noise**: Because the model relies on a single neighbor,
#  it is highly susceptible to noise and outliers in the data.
# This vulnerability is exacerbated by the uniform weighting scheme, 
# which assigns equal importance to all neighbors, regardless of their distance from the query point.

# 3. **High Variance**: The lack of smoothing in the decision boundary,caused by considering only one neighbor, 
# leads to high variance in the model's predictions. 
# . This high variance makes the model unreliable andprone to making erratic predictions.

#Good model:

# Reduced Overfitting: Setting n_neighbors=15 allows the model to consider a larger number of neighbors when making predictions. 
# This helps in reducing the model's sensitivity to individual data points and mitigates the risk of overfitting. 
# By considering more neighbors, the decision boundary becomes smoother and better captures the underlying structure of the data.

# Robustness to Noise: The good model assigns weights to neighbors based on their distance from the
# query point (weights='distance'). Closer neighbors are given higher weights, indicating that they 

# Improved Generalization: With a larger number of neighbors and distance-based weighting, 
# the good model generalizes well to unseen data. It learns the underlying patterns in the data rather than 
# memorizing specific instances from the training set. This leads to better performance on test data and indicates
# that the model has captured the essential features of the dataset.

# Stable Decision Boundary: The decision boundary produced by the good model is 
# smoother and more stable compared to the bad model. It follows the natural structure of the data
# and is less susceptible to fluctuations caused by individual data points. This stability ensures consistent 
# and reliable predictions across different regions of the feature space.
