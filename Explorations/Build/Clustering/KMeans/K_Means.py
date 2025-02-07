#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load and Preprocess Data
df = pd.read_csv("../../../Datasets/penguins.csv")  # Load dataset
features = ['bill_length_mm', 'bill_depth_mm']  # Select features for clustering
X = df[features].dropna()  # Drop missing values for selected features

# Standardize the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Visualize Initial Data
plt.scatter(X['bill_length_mm'], X['bill_depth_mm'], alpha=0.6, c='gray')
plt.title('Initial Data Distribution')
plt.xlabel('Bill Length (mm)')
plt.ylabel('Bill Depth (mm)')
plt.show()

# Apply K-Means Clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Get Clustering Results
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Visualize Clustering Results
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Feature 2 (Standardized)')
plt.legend()
plt.show()

# Evaluate Optimal Number of Clusters (Elbow Method)
wcss = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Curve
plt.plot(range(2, 10), wcss, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares')
plt.show()

# Explore Clustering for Multiple Elbow Numbers
def scatter_elbow(X, X_scaled, n_clusters):
    """
    Visualize clustering results for a given number of clusters.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='rainbow', s=50, alpha=0.7)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', marker='X', s=200, label='Centroids')
    plt.title(f'Clustering with {n_clusters} Clusters')
    plt.xlabel('Feature 1 (Standardized)')
    plt.ylabel('Feature 2 (Standardized)')
    plt.legend()
    plt.show()

# Test Different Cluster Numbers
for n in [3, 4, 5, 6]:
    scatter_elbow(X, X_scaled, n_clusters=n)

