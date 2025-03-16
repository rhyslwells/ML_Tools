import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import datasets, decomposition
from sklearn.preprocessing import scale

# Load Iris dataset
iris = datasets.load_iris()
X, Y = iris.data, iris.target

# Print feature and target names
print("Feature Names:", iris.feature_names)
print("Target Names:", iris.target_names)

# Scale the data
X = scale(X)

# Perform PCA with 3 components
pca = decomposition.PCA(n_components=3)
pca.fit(X)
scores = pca.transform(X)

# Create DataFrame for scores
df_scores = pd.DataFrame(scores, columns=['PC1', 'PC2', 'PC3'])

def map_species_label(y):
    return ['Setosa' if i == 0 else 'Versicolor' if i == 1 else 'Virginica' for i in y]

# Add species labels
df_scores['Species'] = map_species_label(Y)

# Retrieve PCA loadings
df_loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3'], index=iris.feature_names)

# Explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(np.round(np.insert(explained_variance, 0, 0), decimals=3))

# Create DataFrame for variance
df_explained_variance = pd.DataFrame({
    'PC': ['', 'PC1', 'PC2', 'PC3'],
    'Explained Variance': np.insert(explained_variance, 0, 0),
    'Cumulative Variance': cumulative_variance
})

# Plot explained variance
fig1 = px.bar(df_explained_variance, x='PC', y='Explained Variance', text='Explained Variance', width=800)
fig1.update_traces(texttemplate='%{text:.3f}', textposition='outside')
fig1.show()

# Plot cumulative variance and explained variance
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_explained_variance['PC'], y=df_explained_variance['Cumulative Variance'], marker=dict(size=15, color="LightSeaGreen")))
fig2.add_trace(go.Bar(x=df_explained_variance['PC'], y=df_explained_variance['Explained Variance'], marker=dict(color="RoyalBlue")))
fig2.show()

# Separate plots for explained and cumulative variance
fig3 = make_subplots(rows=1, cols=2)
fig3.add_trace(go.Scatter(x=df_explained_variance['PC'], y=df_explained_variance['Cumulative Variance'], marker=dict(size=15, color="LightSeaGreen")), row=1, col=1)
fig3.add_trace(go.Bar(x=df_explained_variance['PC'], y=df_explained_variance['Explained Variance'], marker=dict(color="RoyalBlue")), row=1, col=2)
fig3.show()

# 3D scatter plot of PCA scores
fig4 = px.scatter_3d(df_scores, x='PC1', y='PC2', z='PC3', color='Species')
fig4.show()

# Customized 3D scatter plot
fig5 = px.scatter_3d(df_scores, x='PC1', y='PC2', z='PC3', color='Species', symbol='Species', opacity=0.5)
fig5.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig5.show()

# 3D scatter plot of PCA loadings
fig6 = px.scatter_3d(df_loadings, x='PC1', y='PC2', z='PC3', text=df_loadings.index)
fig6.show()



# other ideas:

# Chapter 5 - Dimensionality Reduction Methods
# Segment 2 - Principal Component Analysis (PCA)
# This script demonstrates PCA applied to the Iris dataset using sklearn's PCA implementation.
# It includes loading the dataset, performing PCA, inspecting explained variance, and visualizing results.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import datasets
from sklearn.decomposition import PCA

# Set up plotting environment
sb.set_style('whitegrid')

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
variable_names = iris.feature_names

# View the first 10 samples of the dataset
print(X[0:10,])

# Perform PCA on the dataset
pca = PCA()
iris_pca = pca.fit_transform(X)

# Output the explained variance ratio for each principal component
print(pca.explained_variance_ratio_)

# Sum of explained variances across all components (should be 1)
print(pca.explained_variance_ratio_.sum())

# Display the principal components (the direction vectors)
comps = pd.DataFrame(pca.components_, columns=variable_names)
print(comps)

# Visualize the principal components with a heatmap
sb.heatmap(comps, cmap="Blues", annot=True)

