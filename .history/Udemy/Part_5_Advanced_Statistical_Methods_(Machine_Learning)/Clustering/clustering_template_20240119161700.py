# Template to: cluster data visualy

# imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans


# Get data

## Load the data
df = pd.read_csv('Countries_exercise.csv')
df.columns

## Plot data


#how intertwined are the clusters.
plt.scatter(data['var1'],data['var2'], c= real_data ['var3'], cmap = 'rainbow')
# Name your axes
plt.xlabel('var1')
plt.ylabel('var2')
plt.show()
## Standarise Variables if necessary

# Import a library which can do that easily
from sklearn import preprocessing
df_scaled = preprocessing.scale(df)
df=df_scaled

## select featrues to cluster

### Encode categoricals if necessary
# df['var1']=df['var1'].map({'element1':0,'element2':1,'element3':2})

# May be more than 2
features=["var1","var2","var3"]
X=df[features]



## How to select number of clusters?

# Use WCSS and elbow method
# number of clusters
wcss=[]
start=
end=
# Create all possible cluster solutions with a loop
for i in range(start,end):
    # Cluster solution with i clusters
    kmeans = KMeans(i)
    # Fit the data
    kmeans.fit(x)
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

Identify the elbow number (there may be more than one thats best)


function to give scatter for each elbow number
    
def scatter_elbow():

## apply clustering with elbow number
num_clusters=elbow_num
kmeans = KMeans(num_clusters)
kmeans.fit(X)

## add cluster information
identified_clusters=kmeans.fit_predict(X)
X['Cluster'] = identified_clusters

## Plot
plt.scatter(data['var1'], data['var2'],c=data_with_clusters['Cluster'], cmap = 'rainbow')
plt.xlim(var1.min,var1.max)
plt.ylim(var1.min,var1.max)
plt.xlabel('var1')
plt.ylabel('var2')
plt.show()

for i in elbow_nums:
    scatter_elbow(i)

