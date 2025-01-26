# Chapter 5 - Dimensionality Reduction Methods
# Segment 1 - Explanatory Factor Analysis

import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn import datasets

# ### Factor Analysis on Iris Dataset

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
variable_names = iris.feature_names

# Preview the first 10 rows of the dataset
print("First 10 rows of the dataset:")
print(pd.DataFrame(X[:10], columns=variable_names))

# Perform Factor Analysis
# You can specify the number of factors using n_components (e.g., n_components=2)
factor = FactorAnalysis(n_components=2, random_state=42)
factor.fit(X)

# Extract and display factor loadings
print("\nFactor Loadings:")
DF = pd.DataFrame(factor.components_, columns=variable_names)
print(DF)

# Additional Information
print("\nExplained Variance:")
print(np.var(factor.transform(X), axis=0))
