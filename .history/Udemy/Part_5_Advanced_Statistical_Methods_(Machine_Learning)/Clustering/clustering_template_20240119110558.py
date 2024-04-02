# Template to: cluster data visualy

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

# imports

# data

plot data scatter


## select featrues to cluster
features=["var1","var2"]
X=df[features]

## apply clustering 

kmeans = KMeans(7)
kmeans.fit(X)

#add cluster information
identified_clusters=kmeans.fit_predict(X)
X['Cluster'] = identified_clusters

plt.scatter(data['var1'], data['var2'],c=data_with_clusters['Cluster'], cmap = 'rainbow')
plt.xlim(var1.min,var1.max)
plt.ylim(var1.min,var1.max)
plt.show()