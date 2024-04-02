# 4. **Predictive Modeling (Optional)**:
#    - If you're interested in predictive analytics, consider building a predictive model to forecast median salary based on features like job classification, location, work type, experience level, and pay period.


#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


"""
Predicting Salary Range: You can predict the median salary ('med_salary') of job postings based on features such as 'title', 'formatted_work_type', 'location', 'formatted_experience_level', 'skill_abr', and 'industry_id'.

Predicting Job Popularity: You can predict the number of applications ('applies') or views ('views') a job posting will receive based on features like 'title', 'company_id', 'formatted_work_type', 'location', 'formatted_experience_level', 'skill_abr', 'industry_id', and whether it's sponsored ('sponsored').

Predicting Remote Work Availability: You can predict whether a job posting will allow remote work ('remote_allowed') based on features such as 'title', 'company_id', 'formatted_work_type', 'location', 'formatted_experience_level', 'skill_abr', and 'industry_id'.

Predicting Job Type: You can predict the type of job ('formatted_work_type') based on features like 'title', 'company_id', 'location', 'formatted_experience_level', 'skill_abr', and 'industry_id'.

Predicting Sponsored Status: You can predict whether a job posting is sponsored ('sponsored') based on features such as 'title', 'company_id', 'location', 'formatted_experience_level', 'skill_abr', and 'industry_id'.

Predicting Job Skill Requirements: You can predict the required skills ('skill_abr') for a job posting based on features like 'title', 'company_id', 'location', 'formatted_experience_level', and 'industry_id'.

Predicting Job Location: You can predict the location ('location') of job postings based on features such as 'title', 'company_id', 'formatted_work_type', 'formatted_experience_level', 'skill_abr', and 'industry_id'.

Predicting Job Experience Level: You can predict the required experience level ('formatted_experience_level') for a job posting based on features like 'title', 'company_id', 'location', 'skill_abr', and 'industry_id'.

"""

#Prediction
# Want to predict median salary based on job classification, location_country, formatted_work_type, and formatted_experience_level,pay_period

# Want a  visual of jobs in this category and their median salary.


# Load preprocessed data
df = pd.read_csv('preprocessed_job_postings.csv')

features_to_keep = 

['job_id', 'company_id', 'title', 'med_salary', 'formatted_work_type',
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
