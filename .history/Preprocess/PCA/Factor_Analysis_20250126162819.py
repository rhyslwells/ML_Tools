# %% [markdown]
# # Chapter 5 - Dimensionality Reduction Methods
# 
# ## Segment 1 - Explanatory factor analysis

# %%
import pandas as pd
import numpy as np

import sklearn
from sklearn.decomposition import FactorAnalysis

from sklearn import datasets

# %% [markdown]
# ### Factor analysis on iris dataset

# %%
iris =  datasets.load_iris()

X = iris.data
variable_names = iris.feature_names

X[0:10,]

# %%
factor = FactorAnalysis().fit(X)

DF = pd.DataFrame(factor.components_, columns=variable_names)
print(DF)

# %%



