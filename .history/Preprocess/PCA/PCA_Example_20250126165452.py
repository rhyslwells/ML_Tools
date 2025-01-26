# Chapter 5 - Dimensionality Reduction Methods
# Segment 2 - Principal Component Analysis (PCA)
# This script demonstrates PCA applied to the Iris dataset using sklearn's PCA implementation.
# It includes loading the dataset, performing PCA, inspecting explained variance, and visualizing results.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import datasets
from sklearn.decomposition import PCA

# Set up plotting environment
%matplotlib inline
plt.rcParams['figure.figsize'] = 5, 4
sb.set_style('whitegrid')

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
variable_names = iris.feature_names

# View the first 10 samples of the dataset
print(X[0:10,])

# Perform PCA on the dataset
pca = PCA()
iris_pca = pca.fit_transform(X)

# Output the explained variance ratio for each principal component
print(pca.explained_variance_ratio_)

# Sum of explained variances across all components (should be 1)
print(pca.explained_variance_ratio_.sum())

# Display the principal components (the direction vectors)
comps = pd.DataFrame(pca.components_, columns=variable_names)
print(comps)

# Visualize the principal components with a heatmap
sb.heatmap(comps, cmap="Blues", annot=True)
