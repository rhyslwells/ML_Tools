#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, roc_auc_score

df = pd.read_csv('preprocessed_job_postings.csv')

features_to_keep = ['job_id', 'company_id', 'title', 'med_salary', 'formatted_work_type',
       'location', 'applies', 'remote_allowed', 'views',
       'formatted_experience_level', 'sponsored', 'skill_abr', 'industry_id']




# Q2: Remote Work Prediction: Which industry is most likely to offer remote work?


# Prepare the Data
# Assuming df_remote is your DataFrame containing relevant features and target variable (remote_allowed)
X_remote = df_remote[['industry_id', 'skill_abr']]
y_remote = df_remote['remote_allowed']

# Encoding categorical variables if necessary

# Model Training
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_remote, y_remote)

# Feature Importance
feature_importance = rf_classifier.feature_importances_
print("Feature Importance:", feature_importance)

#--------------------------------------------------

# Q3: Probability of Remote Work for a Given Company: 
# Is a given company likely to provide remote working (what is the probability of this)?


# Prepare the Data
# Assuming df_company is your DataFrame containing relevant features for the given company
X_company = df_company[['company_id', 'other_features']]
# Encoding categorical variables if necessary

# Prediction
probability_remote_work = rf_classifier.predict_proba(X_company)[:, 1]
print("Probability of Remote Work:", probability_remote_work)






#--------------------------------------------------



