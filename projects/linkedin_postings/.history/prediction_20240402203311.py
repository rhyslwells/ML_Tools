#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv('preprocessed_job_postings.csv')

"""
Predicting Job Location: You can predict the location ('location') of job postings based on features such as 
'title', 'company_id', 'formatted_work_type', 'formatted_experience_level', 'skill_abr', and 'industry_id'.
"""

df["location"].value_counts()
# Get all entries with more than 1 location
location_counts = df["location"].value_counts()
df_multiple_locations = df[df["location"].isin(location_counts[location_counts > 1].index)]
df_multiple_locations['location'].value_counts()
df_multiple_locations.shape
df.shape


features_to_keep = ['job_id', 'company_id', 'title', 'med_salary', 'formatted_work_type',
       'location', 'applies', 'remote_allowed', 'views',
       'formatted_experience_level', 'sponsored', 'skill_abr', 'industry_id']

df.columns

# Select features and target variable
X = df[['job_classification', 'location_country', 'formatted_work_type', 'formatted_experience_level', 'pay_period']]
y = df['median_salary']

# Convert categorical variables into dummy/indicator variables
X = pd.get_dummies(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

"""
Predicting Salary Range: You can predict the median salary ('med_salary') of job postings
 based on features such as 'title', 'formatted_work_type', 'location', 'formatted_experience_level', 'skill_abr', and 'industry_id'.
"""

