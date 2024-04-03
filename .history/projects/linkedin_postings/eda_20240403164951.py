#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import STOPWORDS
import re

#    - Visualize distributions of key features like salary (`median_salary`), 
# experience level (`formatted_experience_level`), job classifications, etc., to understand their characteristics.
#    - Investigate relationships between features, for example, the relationship between job classification and median salary, or between pay period and maximum salary.

# Load data
df = pd.read_csv('preprocessed_job_postings.csv')


# Experience level
# distribution of formatted_experience_level as bar graph
plt.figure(figsize=(10, 6))
sns.countplot(x='formatted_experience_level', data=df)
plt.title('Count of Jobs by Experience Level')
plt.show()

# Distribution of median_salary
plt.figure(figsize=(10, 6))
sns.histplot(df['med_salary'], bins=30, kde=True)
plt.title('Distribution of median_salary')
plt.show()


## Which 'company_id' are in the top 10th percentile of med_salary
top_10_percentile = np.percentile(df['med_salary'], 90)
top_10_companies = df[df['med_salary'] >= top_10_percentile]['company_id'].unique()
print("Company IDs in the top 10th percentile of median salary:", top_10_companies)


# Job title
most_common_titles = df['title'].value_counts().head(10)
print("Top 10 most common job titles:\n", most_common_titles)
top_titles_list = most_common_titles.index.tolist()
print("List of Top 10 most common job titles:\n", top_titles_list)

# What is considered entry level wage for each industry?
'formatted_experience_level', 'industry_id', 'med_salary'

# For a given company how many job postings are available
'company_id', 'job_id'

# Views:
# Some postings have alot more view than others.
# df['views'].value_counts()
df.sort_values(by='views', inplace=False)
df.tail()
#Say that postings outside of 100 views are outliers. How many are there?
df["views"].describe()
df[df["views"]<100].shape
#proposed outliers
df[df["views"]>100].shape
# Reasons for minimal view counts can be reduced time of post on site (explore post time of site)



#Skills 
# Which companies pay the most for the same skills?
def get_top_companies_by_skill(skill, n=5):
    # Filter the DataFrame for rows containing the specified skill
    df_skill = df[df['skill_abr'].str.contains(skill, case=False)]
    
    # Group the DataFrame by 'company_id' and calculate the median salary for each company
    company_salaries = df_skill.groupby('company_id')['med_salary'].median()
    
    # Sort the companies by median salary in descending order
    top_companies = company_salaries.sort_values(ascending=False).head(n)
    
    return top_companies
df['skill_abr'].value_counts()
get_top_companies_by_skill('IT', n=5)



# Additional features






# Group by cluster and calculate median salary for each cluster





-----------------------------------------------
Which skills are most similar pay the most?
'skill_abr', 'med_salary','formatted_experience_level'

If you want to work in this industry the best location is
'industry_id', 'location'



Location Preferences: 
Can we identify regions or cities where certain types of job postings are
 more prevalent?
"""

"""
#random forest:

Given a job and salaray make a prediction on location.

Which industry is most likely to offer remote work?
'industry_id', 'remote_allowed','skill_abr'

"""

"""

#ideas:
#Add another metric like 
# which industries offer remote? (use remote), cluster views and med_salary
# Which companies get the top quartile of views.


# #cluster
# applies vs industry_id

# med_salary vs Industry id


#---------------------------------------------------------------

# Save preprocessed data
# df.to_csv('eda_job_postings.csv', index=False)


#---------------------------------------------------------------


# What are all the attributes we have for a job posting?
# we have two dataframes for agiven job posting.

# # Questions

# ## Which companies post the same job title multiple times? Why would they do this?

# Top 10 most common job titles
most_common_titles = df['title'].value_counts().head(10)
print("Top 10 most common job titles:\n", most_common_titles)


top_titles_list = most_common_titles.index.tolist()

# Display the list of top job titles
print("List of Top 10 most common job titles:\n", top_titles_list)





#extract rows with title

# title = "Sales Manager"
title="Project Manager"
df_title = df[df["title"] == title]


# Group by 'company_id'
grouped_df = df_title.groupby('company_id').size().reset_index(name='count')
sorted_grouped_df = grouped_df.sort_values(by='count', ascending=False)

# Display the grouped DataFrame
print(sorted_grouped_df)


# Initialize an empty dictionary to store the results
companies_with_multiple_posts = {}

for title in top_titles_list:
    df_title = df[df["title"] == title]
    grouped_df = df_title.groupby('company_id').size().reset_index(name='count')
    sorted_grouped_df = grouped_df.sort_values(by='count', ascending=False)
    
    # Extract the company ids for which the count exceeds 5
    companies = sorted_grouped_df[sorted_grouped_df['count'] > 3]['company_id'].tolist()
    
    # Add the result to the dictionary
    companies_with_multiple_posts[title] = companies

# Now, companies_with_multiple_posts is a dictionary where the keys are job titles and the values are lists of company ids
print(companies_with_multiple_posts)


selected_company_id = 73013724.0

# Merge the two DataFrames based on 'company_id'
result_df = pd.merge(df, grouped_df[grouped_df["company_id"] == selected_company_id], on='company_id', how='inner')

# Display the resulting DataFrame
result_df.head()



# is there any duplicate rows after removing identifiers, but with different locations, posted by the same person or different companies?
#Which companies are looking for the same title/job_classification (what constitues the same job)


# ## Analysis with job classifications

#Distrubition of job_classification wrt views.

#number of applications for each job classification compared to the number of views
# or ratio of applications to views fo each class

#Are there certain job_classification associated with median salaries?

# plt.figure(figsize=(12, 8))
# # sns.boxplot(x='job_classification', y='median_salary', data=df)
# plt.xticks(rotation=90)
# plt.title('Boxplot of median Salary by Job Title')
# plt.show()


