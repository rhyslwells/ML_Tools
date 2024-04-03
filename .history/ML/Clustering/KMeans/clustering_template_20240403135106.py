# Template to: cluster data visualy

# imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans


# Get data

#Load the data
df = pd.read_csv('data\Categorical.csv')
df.columns
df.head

# Plot data initially

# How intertwined are the clusters?
# Here we know the continents (see if we can regain them)

### Encode categoricals if necessary
# df['var1']=df['var1'].map({'element1':0,'element2':1,'element3':2})
df['continent_code'] = df['continent'].astype('category').cat.codes

var1, var2, var3 = 'Longitude', 'Latitude', 'continent_code'
plt.scatter(df[var1], df[var2], c=df[var3], cmap='rainbow')
# Name your axes
plt.xlabel(var1)
plt.ylabel(var2)
plt.show()


# Standarise Variables if necessary

# Import a library which can do that easily
from sklearn import preprocessing
df=df[['Longitude', 'Latitude', 'continent_code']]
df_scaled= preprocessing.scale(df) #an array

## select featrues to cluster
# # May be more than 2
# features=["var1","var2","var3"]
# X=df[features]



## How to select number of clusters?

# Use WCSS and elbow method
# number of clusters
wcss=[]
start=2
end=10
# Create all possible cluster solutions with a loop
for i in range(start,end):
    # Cluster solution with i clusters
    kmeans = KMeans(i)
    # Fit the data
    kmeans.fit(df_scaled)
    # Find WCSS for the current iteration
    wcss_iter = kmeans.inertia_
    # Append the value to the WCSS list
    wcss.append(wcss_iter)

# Create a variable containing the numbers from 1 to 6, so we can use it as X axis of the future plot
number_clusters = range(start,end)
# Plot the number of clusters vs WCSS
plt.plot(number_clusters,wcss)
# Name your graph
plt.title('The Elbow Method')
# Name the x-axis
plt.xlabel('Number of clusters')
# Name the y-axis
plt.ylabel('Within-cluster Sum of Squares')

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

'''
Add an extra cluster colomum

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age','Income($)']])
y_predicted


df['cluster']=y_predicted
df.head()

'''




