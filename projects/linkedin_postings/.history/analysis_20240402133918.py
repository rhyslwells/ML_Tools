   
# 3. **Answering Specific Questions**:
#    - Address questions like identifying companies posting the same job title multiple times and reasons behind it. You've started doing this by grouping job titles by company ID.
#    - Analyze job classifications and their relationships with other factors such as views, applications, and median salaries.


#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('job_postings.csv')


# company analysis
# give a company score based on the number of job postings
# ['posting_effectiveness']




# Questions
most_common_titles = df['title'].value_counts().head(10)
print("Top 10 most common job titles:\n", most_common_titles)
top_titles_list = most_common_titles.index.tolist()
print("List of Top 10 most common job titles:\n", top_titles_list)

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

print(companies_with_multiple_posts)



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

print(companies_with_multiple_posts)
