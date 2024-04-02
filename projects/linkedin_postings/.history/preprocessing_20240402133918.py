#!/usr/bin/env python
# coding: utf-8

import pandas as pd

# 1. **Data Exploration and Preprocessing**:
#    - Check for missing values and handle them appropriately. You've already done this for columns like `remote_allowed`, `formatted_experience_level`, and others.
#    - Drop unnecessary columns like `skills_desc` where the majority of the data is missing.
#    - Create new features where necessary, such as `posting_effectiveness` based on the number of applies.


# Load data
df = pd.read_csv('job_postings.csv')



features=['title', 'max_salary', 'med_salary',
       'min_salary', 'pay_period', 'formatted_work_type', 'location',
       'applies', 'remote_allowed', 'views', 'application_url',
       'application_type', 'closed_time', 'formatted_experience_level',
       'skills_desc', 'sponsored', 'work_type']

# Drop unnecessary columns
features_to_drop= features=[ 'application_url','application_type', 'closed_time','compensation_type',  'skills_desc','work_type']
df.drop(features_to_drop, axis=1, inplace=True)

print(df.info())
print(df.count())
print(df.nunique())
missing_percentage = df.isna().mean() * 100
print("Percentage of missing values for each column:",missing_percentage)
# df["sponsored"].value_counts()


# Categorical and Numerical columns inspection
object_feats = df.select_dtypes(include='object').columns
numerical_feats = df.select_dtypes(include='float64').columns
print("Categorical columns:", list(object_feats))
# Categorical columns: ['title', 'pay_period', 'formatted_work_type', 'location', 'formatted_experience_level', 'work_type', 'compensation_type']
print("Numerical columns:", list(numerical_feats))
# Numerical columns: ['company_id', 'max_salary', 'med_salary', 'min_salary', 'applies', 'remote_allowed', 'views']

# to get other features
other_features=['job_id', 'company_id']

#---------------------------------------------------------------

# cleaning categorical columns

df_cat=df[object_feats]
df_cat.columns

# Location (all values filled)
df["location"].value_counts()

#Remote
# fill missing values
df["remote_allowed"].fillna(0, inplace=True)

#Work type
df["formatted_work_type"].value_counts()

# 'title', 
unique_titles_count = df['title'].nunique()
# df["title"].isna().sum() =0

#'formatted_experience_level'
df["formatted_experience_level"].unique()
most_common_experience_level = df['formatted_experience_level'].mode()[0]
#q: how many have na?
number_of_na = df['formatted_experience_level'].isna().sum()
# q: how can we populate those with nan values for formatted_experience_level?
# a: we can fill the missing values with the most common experience level in the dataset.
df['formatted_experience_level'].fillna(most_common_experience_level, inplace=True)

#Pay period #TODO!
df["pay_period"].value_counts()
#count the number of mission values
df["pay_period"].isna().sum()

#---------------------------------------------------------------

# cleaning numerical features


#if med_salary is not present, we can calculate it using the formula (max_salary + min_salary) / 2
# df['med_salary'] = df['med_salary'].fillna((df['max_salary'] + df['min_salary']) / 2)

# if there is only one value in min_salary and max_salary, we can assume that the salary is fixed.
# df['med_salary'] = df[['min_salary', 'max_salary']].mean(axis=1)


#---------------------------------------------------------------


# Save preprocessed data
# df.to_csv('preprocessed_job_postings.csv', index=False)




#add 




    



