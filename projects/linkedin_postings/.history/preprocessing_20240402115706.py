# 1. **Data Exploration and Preprocessing**:
#    - Check for missing values and handle them appropriately. You've already done this for columns like `remote_allowed`, `formatted_experience_level`, and others.
#    - Drop unnecessary columns like `skills_desc` where the majority of the data is missing.
#    - Create new features where necessary, such as `posting_effectiveness` based on the number of applies.

#!/usr/bin/env python
# coding: utf-8

import pandas as pd

# Load data
df = pd.read_csv('job_postings.csv')

df['skills_desc'] 

# Drop unnecessary columns
df.drop(['skills_desc', 'job_id'], axis=1, inplace=True)

# Handle missing values
df['formatted_experience_level'].fillna('Unknown', inplace=True)
df['remote_allowed'].fillna(0, inplace=True)
df['applies'].fillna(0, inplace=True)

# Drop rows with missing salary information
df.dropna(subset=['max_salary', 'min_salary'], inplace=True)

# Add median salary column
df['median_salary'] = (df['max_salary'] + df['min_salary']) / 2

# Save preprocessed data
df.to_csv('preprocessed_job_postings.csv', index=False)
