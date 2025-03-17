# %% [markdown]
# # Chapter 4 - Clustering Models
# ## Segment 1 - K-means method
# ### Setting up for clustering analysis

# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import sklearn
from sklearn.preprocessing import scale
import sklearn.metrics as sm
from sklearn.metrics import confusion_matrix, classification_report

# %%
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets

# %%
plt.figure(figsize=(7,4))

# %%
iris = datasets.load_iris()

# %%

# I want to explore what iris can return:
# dir(iris)
# iris.data

# %%

X = scale(iris.data)
y = pd.DataFrame(iris.target)

# %%
variable_names = iris.feature_names
variable_names 

# %%

X[0:10]

# %% [markdown]
# ## Building and running your model

# %%
clustering = KMeans(n_clusters=3, random_state=5)

clustering.fit(X)

# %% [markdown]
# ## Plotting your model outputs

# %%
iris_df = pd.DataFrame(iris.data)
iris_df.columns
iris_df.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']

# %%
# iris_df.head()

# %%
iris_df.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']
y.columns = ['Targets']

# %%
relabel = np.choose(clustering.labels_, [2, 0, 1]).astype(np.int64)

plt.subplot(1,2,1)

plt.scatter(x=iris_df.Petal_Length, y=iris_df.Petal_Width, c=color_theme[iris.target], s=50)
plt.title('Ground Truth Classification')

plt.subplot(1,2,2)

plt.scatter(x=iris_df.Petal_Length, y=iris_df.Petal_Width, c=color_theme[relabel], s=50)
plt.title('K-Means Classification')

# %% [markdown]
# ## Evaluate your clustering results

# %%
print(classification_report(y, relabel))

# %%



