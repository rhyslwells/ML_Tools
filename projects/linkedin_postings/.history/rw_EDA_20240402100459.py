#!/usr/bin/env python
# coding: utf-8

# # Project description & Problem Definition
# 

# We have some job data scraped from Linkedin. Here we explore the data and try to answer some questions.

## Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('job_postings.csv')
df.head()


# df_comp = pd.read_csv('../data/raw/company_details/companies.csv')
# df_ind = pd.read_csv('../data/raw/company_details/company_industries.csv')
# df_emp = pd.read_csv('../data/raw/company_details/employee_counts.csv')


# # Total features inspection and Initial Preprocessing

# ## Inital inspect

# 
# Drop columns with lots of missing values.
# 
# Can we add values if missing? 

# df.head()
df.columns
# Display the first few rows of the dataframe
df.head()
df.info()


# Categorical  and Numerical columns
# df.info() #df.count()
object_columns = df.select_dtypes(include='object').columns
object_numerical = df.select_dtypes(include='float64').columns


df.nunique()
# df.isnull().sum()
# df.isna().sum()


missing_percentage = df.isna().mean() * 100
print("Percentage of missing values for each column:")
print(missing_percentage)


# those with nas
# df[df.columns[df.isna().sum() > 0].tolist()].info() 
#No na values
df[df.columns[df.isna().sum() == 0].tolist()].info()


# ## Location, work type and remote
# 

# ### location
# 

df["location"].value_counts()


# 
# ### formatted_work_type
# 

df["formatted_work_type"].value_counts()


# ### remote_allowed
# 

df["remote_allowed"].fillna(0, inplace=True)
# Verify the unique values
print(df["remote_allowed"].unique())


# ## Job title, description, level and skills
# 

# ### Title

# Check for missing values
# df['title'].isnull().sum()# 0 missing values

unique_titles_count = df['title'].nunique()
print("\n Number of unique job titles:", unique_titles_count)


# 
# ### Experience_level
# 

# df["formatted_experience_level"].isnull #27% missing values


df["formatted_experience_level"].unique() #['Entry level', 'nan', 'Mid-Senior level', 'Director',
    #    'Associate', 'Executive', 'Internship']
df['formatted_experience_level'].fillna('Unknown', inplace=True)


# distribition of formatted_experience_level as bar graph.

# Bar plot for formatted_experience_level
plt.figure(figsize=(10, 6))
sns.countplot(x='formatted_experience_level', data=df)
plt.title('Count of Jobs by Experience Level')
plt.show()


# Some jobs have an undetermined experience level. We can replace the missing values with "Unknown" to indicate this.
unknown_rows = df[df["formatted_experience_level"] == "Unknown"]
unknown_rows.shape
unknown_rows["title"].value_counts()


# 
# # formatted_work_type and work_type are the same
# 

df["formatted_work_type"].value_counts()


# ### skills_desc
# 

#More than 98% of the data is missing in the "skills_desc" column. We can drop this column.
df.drop("skills_desc",axis=1, inplace=True)


# ## Salary and pay period

## max_salary # 66% missing

## min_salary # 66% missing

df["max_salary"].isnull().sum()

no_salary = df[(df["max_salary"].isnull()) & (df["min_salary"].isnull())]   
no_salary.head()

# Which job posters do not provide sufficent details about the salary?
no_salary


# Which companies should recosider their job postings? ie those that are not effective at attaining applicants.


# Add median salary column
df['median_salary'] = (df['max_salary'] + df['min_salary']) / 2
# df.head()

# Distribution of median_salary
plt.figure(figsize=(10, 6))
sns.histplot(df['median_salary'], bins=30, kde=True)
plt.title('Distribution of median_salary')
plt.show()


# 
# ### pay_period
# 
# 
# 




# 
# ## Linkedin postings
# 

# 
# ### applies
# 

# df["applies"].value_counts()
df["applies"].isnull().sum()
df['applies'].fillna(0, inplace=True)


df['applies'].value_counts()


# We can add a new column to the DataFrame to indicate whether a job posting is effective or not.
# Create a new column 'effectiveness' based on your criteria
df['posting_effectiveness'] = df['applies'].apply(lambda x: 'Bad' if x == 0 else ('Okay' if 0 < x < 10 else ('Effective' if x < 100 else 'Too Many')))

# Display the updated DataFrame
# print(df)
df['posting_effectiveness']


# 
# ### views
# 

# df['views'].fillna(0, inplace=True)
# df["views"].isnull().sum() #7360

# df['views'].value_counts()
df.sort_values(by='views', inplace=False)
df.tail()


#Say that postings outside of 100 views are outliers. How many are there?
df["views"].describe()

df[df["views"]<100].shape
#proposed outliers
df[df["views"]>100].shape


# Assuming df is your DataFrame
plt.boxplot(df[df["views"]<100]["views"])
plt.xlabel('Views')
plt.ylabel('Number of Views')
plt.title('Boxplot of Views')
plt.show()






# Reasons for minimal view counts can be reduced time of post on site.
# can't assume posted at same time.
# Cannot assume poor job posting.


# 
# ### posting_domain
# 

df["posting_domain"].value_counts()

# What domains post the most view/applications jobs?


# ## Job_id and comapny_id

#Job_id

# data\raw\job_details\job_skills.csv # better than description

# job_id and company_id are unique identifiers for each job posting and company, respectively.

# Can ask which companies have multiple job postings.


# Company_id

df_comp.head()

# Do all jobs have a company_id?

# Check for missing values
df['company_id'].isnull().sum()


# company_id
# Check for unique values
df['company_id'].nunique()
df['company_id'].value_counts()

# how popular is a given job classification in a given industry?
# Which industry pays the most for a given job classification?

#What type of job postings are there for each company?

# What industry is popular? view and apps.

# df_comp = pd.read_csv('../data/raw/company_details/companies.csv')
# df_ind = pd.read_csv('../data/raw/company_details/company_industries.csv')
# df_emp = pd.read_csv('../data/raw/company_details/employee_counts.csv')


# # Feature Engineering:
# 

# for feature eng
## Add country col from location



# 
# ## Job classification

# The title is specific, want to distil the type of job that it is. 


# from wordcloud import WordCloud

# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['title']))
# plt.figure(figsize=(12, 6))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.title('Word Cloud of Job Titles')
# # plt.show()



from collections import Counter
from wordcloud import STOPWORDS
import re

# Assuming df is your DataFrame
job_titles = df['title'].dropna()  # Drop any NaN values if present

# Combine all job titles into a single string
all_titles_text = ' '.join(job_titles)

# Remove common English stopwords from the text
stopwords = set(STOPWORDS)
all_titles_text = ' '.join([word for word in all_titles_text.split() if word.lower() not in stopwords])

# Remove symbols and keep only alphanumeric words
all_titles_text = re.sub(r'\W', ' ', all_titles_text)

# Tokenize the text into words
words = all_titles_text.split()

# Count the occurrences of each word
word_counter = Counter(words)

# Get the 100 most common words
most_common_words = word_counter.most_common(100)

# Display the result
print("100 Most Common Words in 'title' column:")
for word, count in most_common_words:
    print(f"{word}: {count}")

# Define the most common words
most_common_words_l=word_list = [word for word, _ in most_common_words]
# most_common_words_l


# Create a new column for classification
df['job_classification'] = ''

# Iterate over each row in the dataframe
for index, row in df.iterrows():
    title = row['title']
    classification = ''
    
    # Check if any of the most common words are present in the title
    for word in most_common_words_l:
        if word in title:
            classification = word
            break
    
    # Assign the classification to the corresponding row
    df.at[index, 'job_classification'] = classification


# Display the first few rows of the dataframe
df.head()


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


# ## Analysis with salary
# 

# Does the highest paying jobs attact the most view and or apps? (scatter pay to views and apps).

# What is the distrubition of applications/views against pay?

# is the pay_period (hrly) indictive of lower paying jobs?
## Boxplot for salary information
# plt.figure(figsize=(12, 8))
# sns.boxplot(x='pay_period', y='max_salary', data=df)
# plt.title('Boxplot of Max Salary by Pay Period')
# plt.show()


#Prediction
# Want to predict median salary based on job classification, location_country, formatted_work_type, and formatted_experience_level,pay_period

# Want a  visual of jobs in this category and their median salary.

