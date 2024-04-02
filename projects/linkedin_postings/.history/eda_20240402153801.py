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
sns.histplot(df['median_salary'], bins=30, kde=True)
plt.title('Distribution of median_salary')
plt.show()



# Job title
most_common_titles = df['title'].value_counts().head(10)
print("Top 10 most common job titles:\n", most_common_titles)
top_titles_list = most_common_titles.index.tolist()
print("List of Top 10 most common job titles:\n", top_titles_list)

# Additional features


# views:
# df['views'].value_counts()
df.sort_values(by='views', inplace=False)
df.tail()
#Say that postings outside of 100 views are outliers. How many are there?
df["views"].describe()
df[df["views"]<100].shape
#proposed outliers
df[df["views"]>100].shape
# Reasons for minimal view counts can be reduced time of post on site (explore post time of site)


# Posting effectiveness
df['posting_effectiveness'] = df['applies'].apply(lambda x: 'Bad' if x == 0 else ('Okay' if 0 < x < 10 else ('Effective' if x < 100 else 'Too Many')))
print(df['posting_effectiveness'])

# which companies have the post effective posts
# company_details\companies.csv



#---------------------------------------------------------------

# Save preprocessed data
# df.to_csv('eda_job_postings.csv', index=False)


#---------------------------------------------------------------


def pull_company_data(company_id):
    # Read employee counts data
    employee_counts = pd.read_csv('company_details/employee_counts.csv')
    employee_counts = employee_counts[employee_counts['company_id'] == company_id]

    # Read company specialties data
    company_specialties = pd.read_csv('company_details/company_specialties.csv')
    company_specialties = company_specialties[company_specialties['company_id'] == company_id]

    # Read company industries data
    company_industries = pd.read_csv('company_details/company_industries.csv')
    company_industries = company_industries[company_industries['company_id'] == company_id]

    # Read companies data
    companies = pd.read_csv('company_details/companies.csv')
    company_info = companies[companies['company_id'] == company_id]

    # Concatenate all dataframes into one
    processed_data = pd.concat([employee_counts, company_specialties, company_industries, company_info], axis=1)

    return processed_data

# Example usage:
# company_id = 12345  # Replace with the desired company ID
# company_data = pull_company_data(company_id)
# print(company_data)


def pull_job_data(job_id):
    #get all the data for a specific job id

    # Read benefits data
    benefits = pd.read_csv('job_details/benefits.csv')
    benefits = benefits[benefits['job_id'] == job_id]

    # Read job industries data
    job_industries = pd.read_csv('job_details/job_industries.csv')
    job_industries = job_industries[job_industries['job_id'] == job_id]

    # Map industry IDs to industry names using industries.csv
    industries = pd.read_csv('maps/industries.csv')
    job_industries = pd.merge(job_industries, industries, on='industry_id', how='left')

    # Read job skills data
    job_skills = pd.read_csv('job_details/job_skills.csv')
    job_skills = job_skills[job_skills['job_id'] == job_id]

    # Map skill abbreviations to skill names using skills.csv
    skills = pd.read_csv('maps/skills.csv')
    job_skills = pd.merge(job_skills, skills, left_on='skill_abr', right_on='skill_abr', how='left')

    # Read salaries data
    salaries = pd.read_csv('job_details/salaries.csv')
    salaries = salaries[salaries['job_id'] == job_id]

    # Concatenate all dataframes into one
    processed_data = pd.concat([benefits, job_industries, job_skills, salaries], axis=1)

    return processed_data


# Example usage:
# job_id = 12345  # Replace with the desired job ID
# job_data = pull_job_data(job_id)
# print(job_data)

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





# result_df.drop("job_id", axis=1, inplace=True)

# # result_df.shape # 63
# result_df.drop_duplicates(inplace=True)
# result_df.shape 


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

#What is considered entry level wage for each job classification?

