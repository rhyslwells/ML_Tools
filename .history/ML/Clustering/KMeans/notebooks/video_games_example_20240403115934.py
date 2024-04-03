#!/usr/bin/env python
# coding: utf-8

# In this notebook I want to implement K-Means clustering algorithm on some simple dataset. I've picked Video Games Sales dataset from Kaggle located under this link: https://www.kaggle.com/rush4ratio/video-game-sales-with-ratings/data. I will try to pick the best games for myself based on their data.

import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
pd.options.mode.chained_assignment = None


# ## Constants

DATA_DIR = "./Video_Games_Sales_as_at_22_Dec_2016.csv"


# ## Loading data

# Reading .csv file

df_raw = pd.read_csv(DATA_DIR)


# Presenting data

print("Samples: " + str(df_raw.shape[0]))
print("Features: " + str(df_raw.shape[1]))


df_raw.head(10)


# ### Data Preprocessing

df_processed = df_raw.copy()


# - Leaving only important columns

COLUMNS_FOR_ANALYSIS = ["Name", "Platform", "Global_Sales", "Critic_Score",
                        "User_Score", "User_Count", "Critic_Count"]
df_processed = df_raw[COLUMNS_FOR_ANALYSIS]


# - Games must be only for consoles that I am using:

PLATFORM_OF_INTEREST = ["3DS", "PC", "PS3", "PS4"]
df_processed = df_raw[df_raw["Platform"].isin(PLATFORM_OF_INTEREST)]


# - Removing samples where Critic Score, Critic Count, User Score, User Cout is nan

df_processed = df_processed.dropna();


# - Creating features that will corellate Critic Score and Critic Count as well as User Score and User Count

df_processed["Critic_Score"] = df_processed["Critic_Score"].apply(float)
df_processed["Critic_Count"] = df_processed["Critic_Count"].apply(float)
df_processed["Critic_Score_Sum"] = df_processed["Critic_Score"] * df_processed["Critic_Count"]


df_processed["User_Score"] = df_processed["User_Score"].apply(float)
df_processed["User_Count"] = df_processed["User_Count"].apply(float)
df_processed["User_Score_Sum"] = df_processed["User_Score"] * df_processed["User_Count"]


# - Drop index

df_processed = df_processed.reset_index(drop=True)


# - Creating input data

df_input = df_processed[["Global_Sales", "User_Score_Sum", "Critic_Score_Sum"]]
df_labels = df_processed[["Name", "Platform"]]


# ## Presenting data

fig = plt.figure(figsize=(16, 16))
ax = fig.add_subplot(111, projection='3d')


ax.scatter(df_input["Critic_Score_Sum"], df_input["User_Score_Sum"], df_input["Global_Sales"])
ax.set_xlabel("Critic Score Sum")
ax.set_ylabel("User Score Sum")
ax.set_zlabel("Global Sales")
plt.show()


# ## K-Means algorithm - raw implementation

class KMeans():
    """Implementation of unsupervised Machine Learning algorithm K-Means used for data 
    clustering. Data is aggregated into 'k' groups. Data sample is assigned to group based
    in it's distance from centroid which is initialized randomly at start and then changed
    by calculating mean of samples assigned to it's group.
    
     Arguments:
    ---------
    k: int
        Number of centroids/data clusters.
    iteration_limit: int
        Maximum numbers of iterations that algorithm can perform in order to reach convergence.     
    """
    @staticmethod
    def _l2_euclidean_distance(p, q):
        """Implementation of similarity function using L2 - Euclidean Distance."""
        return np.sqrt(np.sum(np.square(p - q)))
    
    def __init__(self, k=2, iteration_limit=300):
        self.k = k
        self.iteration_limit = iteration_limit 
        self.centroids = dict()
        self.iter_done = 0
        
    def _init_centroids(self, data):
        """Picking self.k random samples from data and setting them as centroids."""
        samples_num, _ = data.shape
        for i in range(self.k):
            # new centroids are randomly picked samples from inserted data
            self.centroids[i] = data[np.random.randint(0, samples_num)]
    
    def _create_clusters(self, data):
        """Assigning data samples to clusters based on their distance to centroids."""
        
        # init cluster containers with empty lists
        clusters = dict()
        for k in range(self.k):
            clusters.setdefault(k, list())
            
        # get closest cluster to each sample and store it's index in clusters dictionary
        for sample_index, sample in enumerate(data):
            assigned_centroid_index = self._assign_closest_centroid(sample)
            clusters[assigned_centroid_index].append(sample_index)
        return clusters
    
    def _assign_closest_centroid(self, sample):
        """Returning index of closest centroid to inserted sample."""
        closest_centroid_index = 0
        closest_dist = None
        for centroid_index, centroid in self.centroids.items():
            
            # calculating distance of samples from cluster
            dist = self._l2_euclidean_distance(sample, centroid)
            
            # updated closest centroid index based on calculated dist value
            if closest_dist is None:
                closest_dist = dist
                closest_centroid_index = centroid_index
            elif dist < closest_dist:
                closest_centroid_index = centroid_index
                closest_dist = dist
                
        return closest_centroid_index
    
    def _calculate_centroids(self, data, clusters):
        """ Calculate new centroids based on mean of the samples in every cluster."""
        for k in range(self.k):
            # new centroid is an artificial sample created from mean values of features
            # of all samples in cluster
            _, features_num = data.shape
            self.centroids[k] = np.mean(data[clusters[k]], axis=0) if clusters[k] else np.zeros(features_num)        

    def predict(self, data):
        """Assigning indexes of inserted data samples into self.k clusters."""
        
        # randomly initializing clusters
        self._init_centroids(data)
        
        # updating clusters until interation_limit is reached or nothing was changed
        for i in range(self.iteration_limit):
            
            # tracking number of iterations
            self.iter_done = i
            
            clusters = self._create_clusters(data)
            previous_centroids = copy.deepcopy(self.centroids)
            self._calculate_centroids(data, clusters)
            
            # stopping if centroids didn't change
            if all(previous_centroids[k].tolist() == self.centroids[k].tolist() for k in self.centroids.keys()):
                break
                
        return clusters


# ## Data clustering

K = 5


k_means = KMeans(k=K)


clusters = k_means.predict(df_input.values)


for centroid, value in k_means.centroids.items():
    print("Centroid no. " + str(centroid) + ":" + str(value))


print("Clustering finished after " + str(k_means.iter_done) + " iterations.")


game_groups = dict()
for k in range(K):
    game_groups[k] = df_processed.iloc[clusters[k],:]
    print("Group " + str(k) + " samples: " + str(game_groups[k].shape[0]))


fig = plt.figure(figsize=(16, 16))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel("Critic Score Sum")
ax.set_ylabel("User Score Sum")
ax.set_zlabel("Global Sales")

r = lambda: random.randint(0, 255)
COLORS = ["#%02X%02X%02X" % (r(),r(),r()) for _ in range(K)]

plots = list()
labels = list()
for k, df_group in game_groups.items():
    label_ = "Group {}".format(k)
    plot_ = ax.scatter(df_group["Critic_Score_Sum"], df_group["User_Score_Sum"], 
                       df_group["Global_Sales"], c=COLORS[k], label=label_)
    plots.append(plot_)
    labels.append(label_)
    
plt.legend(plots, labels)
plt.show()


# ## Conclusions

def plot_games(df):
    for index, row in df.iterrows():
        print("- " + str(row["Name"] + ", " + str(row["Platform"])))


# - Games with extremely high Critic and User Scores as well as high amount of sales.

plot_games(game_groups[1])


# - Games with best Critic scores, good User Scores and very high sell ratio.

plot_games(game_groups[3])


# - Games which are better than average ones.

plot_games(game_groups[2])


# There is no much info about 3DS games as the console is mostly popular in Japan. This platform should be evaluated separately.
