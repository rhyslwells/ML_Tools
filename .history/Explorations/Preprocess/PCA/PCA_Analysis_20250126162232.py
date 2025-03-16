
# # **Machine Learning in Python: Performing Principal Component Analysis (PCA)**
# 
# Chanin Nantasenamat
# 
# <i>Data Professor YouTube channel, http://youtube.com/dataprofessor </i>
# 
# In this Jupyter notebook, we will be performing Principal Component Analysis (PCA) using the Iris data set as an example.


# ---


# ## **1. Iris data set**


# ### Load library


from sklearn import datasets


# ### Load dataset


iris = datasets.load_iris()


# ### Input features


print(iris.feature_names)


# ### Output features


print(iris.target_names)


# ### Assigning Input (X) and Output (Y) variables
# Let's assign the 4 input variables to X and the output variable (class label) to Y


X = iris.data
Y = iris.target


# ### Let's examine the data dimension


X.shape


Y.shape


# ---


# ## **2. PCA analysis**


# ### 2.1. Load library


from sklearn.preprocessing import scale # Data scaling
from sklearn import decomposition #PCA
import pandas as pd # pandas


# ### 2.2. Data scaling


X = scale(X)


# ### 2.3. Perform PCA analysis


# Here we define the number of PC to use as 3


pca = decomposition.PCA(n_components=3)
pca.fit(X)



# #### 2.4. Compute and retrieve the **scores** values


scores = pca.transform(X)


scores_df = pd.DataFrame(scores, columns=['PC1', 'PC2', 'PC3'])
scores_df


Y_label = []

for i in Y:
  if i == 0:
    Y_label.append('Setosa')
  elif i == 1:
    Y_label.append('Versicolor')
  else:
    Y_label.append('Virginica')

Species = pd.DataFrame(Y_label, columns=['Species'])


df_scores = pd.concat([scores_df, Species], axis=1)


# #### 2.5. Retrieve the **loadings** values


loadings = pca.components_.T
df_loadings = pd.DataFrame(loadings, columns=['PC1', 'PC2','PC3'], index=iris.feature_names)
df_loadings


# #### 2.6. **Explained variance** for each PC


explained_variance = pca.explained_variance_ratio_
explained_variance


# ## **3. Scree Plot**


# ### 3.1. Import library


import numpy as np
import plotly.express as px


# ### 3.2. Preparing explained variance and cumulative variance


# #### 3.2.1. Preparing the explained variance data


explained_variance


explained_variance = np.insert(explained_variance, 0, 0)


# #### 3.2.2. Preparing the cumulative variance data


cumulative_variance = np.cumsum(np.round(explained_variance, decimals=3))


# #### 3.2.3. Combining the dataframe


pc_df = pd.DataFrame(['','PC1', 'PC2', 'PC3'], columns=['PC'])
explained_variance_df = pd.DataFrame(explained_variance, columns=['Explained Variance'])
cumulative_variance_df = pd.DataFrame(cumulative_variance, columns=['Cumulative Variance'])


df_explained_variance = pd.concat([pc_df, explained_variance_df, cumulative_variance_df], axis=1)
df_explained_variance


# #### 3.2.4. Making the scree plot


# ##### 3.2.4.1. Explained Variance


# https://plotly.com/python/bar-charts/

fig = px.bar(df_explained_variance, 
             x='PC', y='Explained Variance',
             text='Explained Variance',
             width=800)

fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
fig.show()


# ##### 3.2.4.2. Explained Variance + Cumulative Variance


# https://plotly.com/python/creating-and-updating-figures/

import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df_explained_variance['PC'],
        y=df_explained_variance['Cumulative Variance'],
        marker=dict(size=15, color="LightSeaGreen")
    ))

fig.add_trace(
    go.Bar(
        x=df_explained_variance['PC'],
        y=df_explained_variance['Explained Variance'],
        marker=dict(color="RoyalBlue")
    ))

fig.show()


# ##### 3.2.4.3. Explained Variance + Cumulative Variance (Separate Plot)


from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(rows=1, cols=2)

fig.add_trace(
    go.Scatter(
        x=df_explained_variance['PC'],
        y=df_explained_variance['Cumulative Variance'],
        marker=dict(size=15, color="LightSeaGreen")
    ), row=1, col=1
    )

fig.add_trace(
    go.Bar(
        x=df_explained_variance['PC'],
        y=df_explained_variance['Explained Variance'],
        marker=dict(color="RoyalBlue"),
    ), row=1, col=2
    )

fig.show()


# ## **4. Scores Plot**
# 
# Source: https://plotly.com/python/3d-scatter-plots/


# ### 4.1. Load library
# [API Documentation](https://plotly.com/python-api-reference/plotly.express.html) for *plotly.express* package


import plotly.express as px


# ### 4.2. Basic 3D Scatter Plot


fig = px.scatter_3d(df_scores, x='PC1', y='PC2', z='PC3',
              color='Species')

fig.show()


# ### 4.3. Customized 3D Scatter Plot


fig = px.scatter_3d(df_scores, x='PC1', y='PC2', z='PC3',
              color='Species',
              symbol='Species',
              opacity=0.5)

# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

# https://plotly.com/python/templates/
#fig.update_layout(template='plotly_white') # "plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"


# ## **5. Loadings Plot**


loadings_label = df_loadings.index
# loadings_label = df_loadings.index.str.strip(' (cm)')

fig = px.scatter_3d(df_loadings, x='PC1', y='PC2', z='PC3',
                    text = loadings_label)

fig.show()


# ---


