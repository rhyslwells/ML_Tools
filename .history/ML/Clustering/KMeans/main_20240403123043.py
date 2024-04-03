#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

#vis
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.colors import ListedColormap
import seaborn as sns

#data
import statsmodels.api as sm

#modeling
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans


# 
# ## Penguins (for clusters)
# 

penguins= pd.read_csv("../../data/penguins.csv")
penguins.head()
# penguins.shape


# ### Sklearn

# Apply K-means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(x)

# Get cluster centers and labels
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Plot the data points and cluster centers
plt.figure(figsize=(6,4))
plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Bill Length (mm)')
plt.ylabel('Bill Depth (mm)')
plt.legend()
plt.show()


sns.pairplot(
    penguins,
    x_vars=["bill_length_mm"],
    y_vars=["bill_depth_mm"],hue="species")


# ----------

'''
Load data (from sklearn datasets)

Scale data if needed

Determine X (input), y (target)
(you are clustering with respect to y?)

Build modle:
clustering = KMeans(n_clusters=3, random_state=5)
clustering.fit(X)



Then select two features the inital dataset, then using  clustering (as mapping) can
color into groups.

You have already clusted 

then evaluate with

print(classification_report(y, relabel))

'''

#-------------------



