
# # Cluster analysis
# In this notebook we explore heatmaps and dendrograms


# ## Import the relevant libraries

import numpy as np
import pandas as pd
import seaborn as sns
# We don't need matplotlib this time


# ## Load the data


# Load the standardized data
# index_col is an argument we can set to one of the columns
# this will cause one of the Series to become the index
data = pd.read_csv('Country clusters standardized.csv', index_col='Country')


# Create a new data frame for the inputs, so we can clean it
x_scaled = data.copy()
# Drop the variables that are unnecessary for this solution
x_scaled = x_scaled.drop(['Language'],axis=1)


# Check what's inside
x_scaled


# ## Plot the data


# Using the Seaborn method 'clustermap' we can get a heatmap and dendrograms for both the observations and the features
# The cmap 'mako' is the coolest if you ask me
sns.clustermap(x_scaled, cmap='mako')


