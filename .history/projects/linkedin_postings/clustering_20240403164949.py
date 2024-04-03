# There are no clear clusters.

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

#---------------------------------------------------
"""What are the top 5 companies with similar job postings to a 
given company, based on features such as 'industry_id', 
'skill_abr', 'title', 'job_id', and 'formatted_experience_level'?"""

from sklearn.cluster import KMeans

# Prepare data
features_interest = ['company_id', 'title', 'formatted_experience_level', 'skill_abr']
df_company_data = df[features_interest]

# Encode categorical variables
label_encoder = LabelEncoder()
for feature in features_interest:
       if df_company_data[feature].dtype == 'object':
              df_company_data[feature] = label_encoder.fit_transform(df_company_data[feature])


# Standardize features
scaler = StandardScaler()
df_company_data = scaler.fit_transform(df_company_data)

# Apply KMeans clustering
k = 5  # Choose the number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(df_company_data)

# Find similar companies to a given company
# Assume 'given_company_id' is the company for which we want to find similar companies
given_company_id = 92699700      # Example company ID
given_company_cluster = kmeans.predict(df_company_data[df_company_data['company_id'] == given_company_id])

# Get companies in the same cluster as the given company
similar_companies_indices = df_company_data[kmeans.labels_ == given_company_cluster[0]].index

# Exclude the given company itself from the list
similar_companies_indices = similar_companies_indices[similar_companies_indices != given_company_id]

# Get the top 5 similar companies based on some similarity metric (e.g., Euclidean distance)
# ... (Code for ranking similar companies)

# Output the list of top 5 similar companies
top_5_similar_companies = df.loc[similar_companies_indices].head(5)
print(top_5_similar_companies)

#---------------------------------------------------

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

