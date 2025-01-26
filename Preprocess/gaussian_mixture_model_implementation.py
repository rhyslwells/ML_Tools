# [Script Title]: Comparing GMM Covariance Types and k-Means Clustering
# Purpose: Demonstrate how GMM handles non-spherical distributions using different covariance types.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

# Step 1: Generate synthetic data
np.random.seed(42)
n_samples = 300

# Generate three clusters with different shapes
cluster_1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 4]], size=n_samples)
cluster_2 = np.random.multivariate_normal([6, 0], [[2, 1.5], [1.5, 2]], size=n_samples)
cluster_3 = np.random.multivariate_normal([3, 6], [[3, -1], [-1, 1]], size=n_samples)
data = np.vstack([cluster_1, cluster_2, cluster_3])

# Step 2: Fit Gaussian Mixture Models with different covariance types
covariance_types = ['spherical', 'diag', 'tied', 'full']
gmm_results = []

for cov_type in covariance_types:
    gmm = GaussianMixture(n_components=3, covariance_type=cov_type, random_state=0)
    gmm.fit(data)
    labels = gmm.predict(data)
    gmm_results.append((cov_type, labels))

# Step 3: Fit k-Means
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans_labels = kmeans.fit_predict(data)

# Step 4: Visualize the results
fig, ax = plt.subplots(1, 5, figsize=(25, 5))

# k-Means visualization
ax[0].scatter(data[:, 0], data[:, 1], c=kmeans_labels, cmap='viridis', marker='o', edgecolor='k', s=50)
ax[0].set_title('k-Means Clustering')
ax[0].set_xlabel('Feature 1')
ax[0].set_ylabel('Feature 2')
ax[0].grid()

# GMM visualizations for each covariance type
for i, (cov_type, labels) in enumerate(gmm_results):
    ax[i + 1].scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)
    ax[i + 1].set_title(f'GMM ({cov_type.capitalize()} Covariance)')
    ax[i + 1].set_xlabel('Feature 1')
    ax[i + 1].set_ylabel('Feature 2')
    ax[i + 1].grid()

plt.tight_layout()
plt.show()
