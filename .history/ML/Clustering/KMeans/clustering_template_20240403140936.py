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

from sklearn import preprocessing


    
# Load data (from sklearn datasets)
df = pd.read_csv("../../data/penguins.csv")

# initial scatterplot: How intertwined are the clusters?


# Standarise Variables if necessary
df_scaled= preprocessing.scale(df) #an array

# Select featrues to cluster (May be more than 2)
features=['bill_length_mm', 'bill_depth_mm']
X=df[features]
x = df[features]
# determine X (input), y (target)
# (you are clustering with respect to y?)



# Apply K-means clustering with k=3 (see elbow method below)
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
# sns.pairplot(
#     df,
#     x_vars=["bill_length_mm"],
#     y_vars=["bill_depth_mm"],hue="species")

#make prediction

# Add an extra cluster colomum
y_predicted = kmeans.fit_predict(df[['Bill Length (mm)']])
y_predicted
df['cluster']=y_predicted
df.head()

# Then select two features the inital dataset, then using  clustering (as mapping) can
# color into groups.

#Evaluate with
# print(classification_report(y, relabel))




#--------------------------------------------------------------------------------
# How to select number of clusters?
# use WCSS and elbow method to select the number of clusters
wcss = []
start, end = 2, 10

for i in range(start, end):
    kmeans = KMeans(i)
    kmeans.fit(df_scaled)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)

number_clusters = range(start, end)
plt.plot(number_clusters, wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster Sum of Squares')
plt.show()

# Identify the elbow numbers (there may be more than one thats best)
elbow_nums=[4,5,6,7,8]
# function to give scatter for each elbow number
def scatter_elbow(X, elbow_num, var1, var2):
    """
    Apply clustering with elbow method and plot a scatter plot with cluster information.

    Parameters:
    - X: DataFrame, input data for clustering
    - elbow_num: int, number of clusters determined by elbow method
    - var1, var2: str, names of the variables for the scatter plot

    Returns:
    None (plots the scatter plot)
    """
    # Apply clustering with elbow number
    kmeans = KMeans(elbow_num)
    kmeans.fit(X)

    # Add cluster information
    identified_clusters = kmeans.fit_predict(X)
    X['Cluster'] = identified_clusters

    # Plot
    plt.scatter(X[var1], X[var2], c=X['Cluster'], cmap='rainbow')
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.title(f"{elbow_num}-Clustering for {var1}-{var2}")
    plt.show()

# Example usage:
# scatter_elbow(data, elbow_num, 'var1', 'var2')


for elbow_num in elbow_nums:
    scatter_elbow(df, elbow_num, var1, var2)
#--------------------------------------------------------------------------------


