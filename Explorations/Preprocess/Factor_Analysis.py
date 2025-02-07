# Chapter 5 - Dimensionality Reduction Methods
# Segment 1 - Explanatory Factor Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis
from sklearn import datasets

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target  # Target labels (species)
variable_names = iris.feature_names
target_names = iris.target_names  # Names of the species

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

# Visualize the factors in the new latent space (Factor 1 vs Factor 2)
factor_transformed = factor.transform(X)

plt.figure(figsize=(8, 6))

# Scatter plot colored by species
for i, species in enumerate(target_names):
    plt.scatter(factor_transformed[y == i, 0], factor_transformed[y == i, 1], label=species, alpha=0.6)

plt.title('Factor Analysis - 2D Latent Space')
plt.xlabel('Factor 1')
plt.ylabel('Factor 2')
plt.legend()
plt.show()

# Explore the relationship between the factors and the target classes (species)
# This will display the average factor values for each species
species_factor_means = pd.DataFrame(factor_transformed, columns=['Factor 1', 'Factor 2'])
species_factor_means['Species'] = [target_names[i] for i in y]

print("\nAverage Factor Values by Species:")
print(species_factor_means.groupby('Species').mean())
