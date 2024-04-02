#Clustering with KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

"""
For clustering, you can use only numerical features

Apply KNN Algorithm: Once the data is prepared and features are selected, 
you can apply the KNN algorithm to cluster the data points. 
You'll have to choose the number of clusters (k) beforehand.
"""

df = pd.read_csv('preprocessed_job_postings.csv')


# 1. Prepare the data - Assume df is your DataFrame
df.columns

features_interest =['job_id', 'company_id','title', 'med_salary', 'formatted_work_type',
       'location', 'applies', 'remote_allowed', 'views',
       'formatted_experience_level', 'sponsored', 'skill_abr', 'industry_id']


# 2. Select features #TODO! Which ones to choose?

df['formatted_work_type'].value_counts() # so weights could be dropped.
df['formatted_experience_level'].value_counts() # so weights could be dropped.


"""

Which comapanies pay the most for the same skills?
'company_id', 'med_salary', 'formatted_experience_level', 'skill_abr'

If you are interested in this company you might like this one too
'company_id', 'industry_id','skill_abr'

Which companies are most similar in terms of job postings?
'company_id', 'title', 'job_id', 'formatted_experience_level', 'skill_abr'

-----------------------------------------------
Which skills are most similar pay the most?
'skill_abr', 'med_salary','formatted_experience_level'

If you want to work in this industry the best location is
'industry_id', 'location'

Which industry is most likely to offer remote work?
'industry_id', 'remote_allowed','skill_abr'

Location Preferences: 
Can we identify regions or cities where certain types of job postings are more prevalent?


"""


df_new = df[features_interest]


#1.1 turn categorical variables into numerical





# 3. Normalize features to same scale.
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_new)

# 4. Apply KNN Algorithm
# Let's choose number of clusters, k=3 for example
k = 3
knn_model = NearestNeighbors(n_neighbors=k)
knn_model.fit(scaled_features)

# Get clusters
distances, indices = knn_model.kneighbors(scaled_features)

# You can now use these clusters for analysis or visualization
