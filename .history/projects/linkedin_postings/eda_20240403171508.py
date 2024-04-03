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

# Which skills are most similar pay the most?
# Group by skill and calculate average salary
skill_salary = df.groupby('skill_abr')['med_salary'].mean().reset_index()
# Rank skills based on average salary
ranked_skills = skill_salary.sort_values(by='med_salary', ascending=False)
# Output the ranked skills
print(ranked_skills.head(10))  # Display the top 10 skills with the highest average salary



#location
#determine the best location for a specific industry,
def get_best_locations_for_industry(df, industry_id):
    """
    Get the best locations for a specific industry based on average salary.

    Parameters:
    - df (DataFrame): Input DataFrame containing job posting data
    - industry_id (int): Industry ID for which to find the best locations

    Returns:
    - DataFrame: Ranked locations for the specified industry based on average salary
    """

    # Filter data for the specified industry
    industry_data = df[df['industry_id'] == industry_id]

    # Group by industry and location, and calculate average salary
    industry_location_avg_salary = industry_data.groupby('location')['med_salary'].mean().reset_index()
    # industry_location_avg_salary.rename(columns={'med_salary': 'avg_salary'}, inplace=True)

    # Rank locations based on average salary
    ranked_locations = industry_location_avg_salary.sort_values(by='med_salary', ascending=False)

    return ranked_locations
# Call the function to get the ranked locations for industry with ID 123
industry_id = 135
ranked_locations = get_best_locations_for_industry(df, industry_id)
print(f"The best location for industry {industry_id} in terms of med_salary is:\n", ranked_locations.head(5))  # Display the top 10 locations for the specified industry



#What location are there the most job postings for 'instustry_id' and skill_abr=['ENG','IT'] are
def identify_prevalent_locations_with_criteria(df, industry_id, skills):
    """
    Identify regions where job postings for a specific industry and skills are more prevalent.

    Parameters:
    - df (DataFrame): Input DataFrame containing job posting data
    - industry_id (int): Industry ID for which to filter job postings
    - skills (list): List of skills to filter job postings

    Returns:
    - DataFrame: Ranked list of regions where job postings for the specified industry and skills are more prevalent
    """

    # Filter data based on industry and skills
    filtered_data = df[(df['industry_id'] == industry_id) & df['skill_abr'].isin(skills)]

    # Group by location and count the number of job postings
    location_job_postings = filtered_data.groupby('location')['job_id'].count().reset_index()
    location_job_postings.rename(columns={'job_id': 'num_job_postings'}, inplace=True)

    # Rank locations based on the number of job postings
    ranked_locations = location_job_postings.sort_values(by='num_job_postings', ascending=False)

    return ranked_locations

# Call the function to identify prevalent locations for a specific industry and skills
industry_id = 135  # Example industry ID
skills = ['ENG', 'IT']  # Example list of skills
prevalent_locations = identify_prevalent_locations_with_criteria(df, industry_id, skills)
print(prevalent_locations.head(5))  # Display the top 10 locations for the specified industry and skills
# Group by industry_id and skills_abr
tx = df[df['location'] == 'Houston, TX']
tx_grouped = tx.groupby(['industry_id', 'skill_abr'])
#get the group that has industry_id=135 and skill_abr ENG
tx_grouped.get_group((135, 'ENG')).shape
tx_grouped.get_group((135, 'IT')).shape



#using random forest:
Given an industry, skill_abr, location, make a prediction on the salaray



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


