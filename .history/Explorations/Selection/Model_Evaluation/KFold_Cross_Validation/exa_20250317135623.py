# %%
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# %%
iris = load_iris()

# %% [markdown]
# **Logistic Regression**

# %%
l_scores = cross_val_score(LogisticRegression(), iris.data, iris.target)
l_scores

# %%
np.average(l_scores)

# %% [markdown]
# **Decision Tree**

# %%
d_scores = cross_val_score(DecisionTreeClassifier(), iris.data, iris.target)
d_scores

# %%
np.average(d_scores)

# %% [markdown]
# **Support Vector Machine (SVM)**

# %%
s_scores = cross_val_score(SVC(), iris.data, iris.target)
s_scores

# %%
np.average(s_scores)

# %% [markdown]
# **Random Forest**

# %%
r_scores = cross_val_score(RandomForestClassifier(n_estimators=40), iris.data, iris.target)
r_scores

# %%
np.average(r_scores)

# %% [markdown]
# **Best score so far is from SVM: 0.97344771241830064**


