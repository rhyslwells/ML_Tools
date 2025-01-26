# [Script Title]: Programmatic Exploration of Gaussian Mixture Models  
# Purpose: Demonstrating the implementation and application of Gaussian Mixture Models for clustering  
#  
# Step 1: Import necessary libraries  
# Step 2: Generate synthetic data for demonstration  
# Step 3: Fit a Gaussian Mixture Model to the data  
# Step 4: Visualize the results  
#  
# Introductory Example:  
# This example generates synthetic data and fits a GMM to illustrate how GMMs can cluster data points based on their distributions.  

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Step 1: Generate synthetic data
np.random.seed(0)
n_samples = 300
# Create two clusters of data
cluster_1 = np.random.normal(loc=0, scale=1, size=(n_samples, 2))
cluster_2 = np.random.normal(loc=5, scale=1, size=(n_samples, 2))
data = np.vstack([cluster_1, cluster_2])

# Step 2: Fit a Gaussian Mixture Model
gmm = GaussianMixture(n_components=2, covariance_type='full')
gmm.fit(data)

# Step 3: Predict cluster memberships
labels = gmm.predict(data)

# Step 4: Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.title('Gaussian Mixture Model Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid()
plt.show()