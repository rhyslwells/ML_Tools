# 4. **Predictive Modeling (Optional)**:
#    - If you're interested in predictive analytics, consider building a predictive model to forecast median salary based on features like job classification, location, work type, experience level, and pay period.


#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


#Prediction
# Want to predict median salary based on job classification, location_country, formatted_work_type, and formatted_experience_level,pay_period

# Want a  visual of jobs in this category and their median salary.


# Load preprocessed data
df = pd.read_csv('preprocessed_job_postings.csv')

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
