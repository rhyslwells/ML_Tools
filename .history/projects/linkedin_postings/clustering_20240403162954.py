#Clustering with KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns


"""
For clustering, you can use only numerical features

Apply KNN Algorithm: Once the data is prepared and features are selected, 
you can apply the KNN algorithm to cluster the data points. 
You'll have to choose the number of clusters (k) beforehand.
"""

df = pd.read_csv('preprocessed_job_postings.csv')


# 1. Prepare the data - Assume df is your DataFrame
# df.columns
features_interest=['job_id', 'company_id','med_salary',
       'location', 'skill_abr','applies', 'industry_id']


df_new = df[features_interest]

#1.1 turn categorical variables into numerical
# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Iterate over each categorical feature and encode it
for feature in features_interest:
       if df_new[feature].dtype == 'object':
              df_new[feature] = label_encoder.fit_transform(df_new[feature])

df_new.head()

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_new)


# plot as scatter graph medi


# 4. Apply KNN Algorithm
# Let's choose number of clusters, k=3 for example
k = 3
knn_model = KMeans(k)
knn_model.fit(scaled_features)

# 5. Add cluster labels to DataFrame
df_new['Cluster'] = knn_model.labels_

# 6. Plot the data points and cluster centers
plt.scatter(df_new['skill_abr'], df_new['med_salary'], c=df_new['Cluster'], cmap='plasma', s=50, alpha=0.7)
#give a better cmap

