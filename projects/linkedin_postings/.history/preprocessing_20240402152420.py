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

#ensure job_id and company_id are integers
df['job_id'] = df['job_id'].astype(int)
df['company_id'] = df['company_id'].astype(int)


# Drop rows with missing company_id
df.dropna(subset=['company_id'], inplace=True)
# df['company_id'].isna().sum()=0

#---------------------------------------------------------------

# cleaning categorical columns

df_cat=df[object_feats]
df_cat.columns

# Location (all values filled)
df["location"].value_counts()



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
df['formatted_experience_level'].value_counts()


#Pay period #TODO!

df["pay_period"].value_counts()
#count the number of mission values
df["pay_period"].isna().sum()

# get the term that has pay period 'ONCE'
once = df[df['pay_period'] == 'ONCE']
#drop this term
df = df[df['pay_period'] != 'ONCE']



df["pay_period"].isna().sum()
# Filter rows with missing pay_period
na_pay_period_df = df[df["pay_period"].isna()]
na_pay_period_df.head()


# convert all to year 
# if it is monthly multiply by 12
df.loc[df['pay_period'] == 'MONTH', 'max_salary'] *= 12
df.loc[df['pay_period'] == 'MONTH', 'min_salary'] *= 12
df.loc[df['pay_period'] == 'MONTH', 'med_salary'] *= 12
#replace the pay period with year

# if it is weekly multiply by 52
df.loc[df['pay_period'] == 'WEEK', 'max_salary'] *= 52
df.loc[df['pay_period'] == 'WEEK', 'min_salary'] *= 52
df.loc[df['pay_period'] == 'WEEK', 'med_salary'] *= 52
#replace the pay period with year

# if it is hourly multiply by 2080
df.loc[df['pay_period'] == 'HOUR', 'max_salary'] *= 2080
df.loc[df['pay_period'] == 'HOUR', 'min_salary'] *= 2080
df.loc[df['pay_period'] == 'HOUR', 'med_salary'] *= 2080


# how can we determine the pay_period for those that are missing?

#First idea: Fail: these terms do not have any salary data and not contained in 'job_details/salaries.csv'.##
# Second: Fail: Based on the industry, we can determine the pay_period for the missing values using the mode.
# Third (Focus): Based on the industry we can get the average med_salary and bypass using pay_period (as is the end goal).
#Then we can drop max_salary, min_salary, and pay_period columns.

# now drop the pay_period column
df.drop('pay_period', axis=1, inplace=True)

# industry_id

# for df based on job_id with job_details\job_industries.csv get industry_id.
# Read job_industries data
job_industries = pd.read_csv('job_details/job_industries.csv')
# Merge df with job_industries to get the corresponding industry_id
df = pd.merge(df, job_industries[['job_id', 'industry_id']], on='job_id', how='inner')
df.head()

df['med_salary']=df['min_salary']+df['max_salary']/2

# now group by industry_id and get the average min_salary, max_salary



#---------------------------------------------------------------

# cleaning numerical features

df_num=df[numerical_feats]
df_num.columns

#Remote #DONE
# fill missing values
df["remote_allowed"].fillna(0, inplace=True)

#'views' #DONE
df["views"].value_counts()
#count the number of mission values
df["views"].isna().sum()
#replace those missing with 0
df["views"].fillna(0, inplace=True)
df["views"].isna().sum()

#'applies' #DONE
df["applies"].value_counts()
df["applies"].isna().sum()
df["applies"].fillna(0, inplace=True)
df["applies"].isna().sum()

#salaries




# determine how many entries have missing values for max_salary, min_salary, and med_salary
missing_max_salary = df['max_salary'].isna().sum()
missing_min_salary = df['min_salary'].isna().sum()
missing_med_salary = df['med_salary'].isna().sum()

print("Number of missing values for max_salary:", missing_max_salary)
print("Number of missing values for min_salary:", missing_min_salary)
print("Number of missing values for med_salary:", missing_med_salary)

#if med_salary is not present, we can calculate it using the formula (max_salary + min_salary) / 2
# df['med_salary'] = df['med_salary'].fillna((df['max_salary'] + df['min_salary']) / 2)

# if there is only one value in min_salary and max_salary, we can assume that the salary is fixed.
# df['med_salary'] = df[['min_salary', 'max_salary']].mean(axis=1)

# based on the pay period we can calculate the missing values for max_salary, min_salary, and med_salary



df['pay_period'].value_counts()





# determine the entries with missing values for max_salary, min_salary, and med_salary
missing_entries = df[df['max_salary'].isna() & df['min_salary'].isna() & df['med_salary'].isna()]
print("Entries with missing values for max_salary, min_salary, or med_salary:")
print(missing_entries.shape)



#'max_salary'
df["max_salary"].isna().sum()

# 'med_salary', 
df["med_salary"].isna().sum()

# 'min_salary'
df["min_salary"].isna().sum()








#---------------------------------------------------------------


# Save preprocessed data
# df.to_csv('preprocessed_job_postings.csv', index=False)




#add 




    



