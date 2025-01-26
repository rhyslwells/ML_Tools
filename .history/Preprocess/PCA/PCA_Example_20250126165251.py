# %% [markdown]
# # Chapter 5 - Dimensionality Reduction Methods
# ## Segment 2 - Principal component analysis (PCA)

# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import pylab as plt
import seaborn as sb
from IPython.display import Image
from IPython.core.display import HTML 
from pylab import rcParams

import sklearn
from sklearn import datasets

# %%
from sklearn import decomposition
from sklearn.decomposition import PCA

# %%
%matplotlib inline
rcParams['figure.figsize'] = 5, 4
sb.set_style('whitegrid')

# %% [markdown]
# ### PCA on the iris dataset

# %%
iris = datasets.load_iris()
X = iris.data
variable_names = iris.feature_names

X[0:10,]

# %%
pca = decomposition.PCA()
iris_pca = pca.fit_transform(X)

pca.explained_variance_ratio_

# %%
pca.explained_variance_ratio_.sum()

# %%
comps = pd.DataFrame(pca.components_, columns=variable_names)
comps

# %%
sb.heatmap(comps, cmap="Blues", annot=True)

# %%



