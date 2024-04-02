# 2. **Analysis and Visualization**:
#    - Visualize distributions of key features like salary (`median_salary`), experience level (`formatted_experience_level`), job classifications, etc., to understand their characteristics.
#    - Investigate relationships between features, for example, the relationship between job classification and median salary, or between pay period and maximum salary.

#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import STOPWORDS
import re

# Load data
df = pd.read_csv('job_postings.csv')

# Initial inspection
print(df.columns)
print(df.head())
print(df.info())

# Categorical and Numerical columns inspection
object_columns = df.select_dtypes(include='object').columns
object_numerical = df.select_dtypes(include='float64').columns
print(df.nunique())
missing_percentage = df.isna().mean() * 100
print("Percentage of missing values for each column:")
print(missing_percentage)

# Location, work type and remote
print(df["location"].value_counts())
print(df["formatted_work_type"].value_counts())
df["remote_allowed"].fillna(0, inplace=True)
print(df["remote_allowed"].unique())

# Job title, description, level and skills
unique_titles_count = df['title'].nunique()
print("\n Number of unique job titles:", unique_titles_count)
print(df["formatted_experience_level"].unique())

# Distribution of formatted_experience_level as bar graph
plt.figure(figsize=(10, 6))
sns.countplot(x='formatted_experience_level', data=df)
plt.title('Count of Jobs by Experience Level')
plt.show()

# Drop unnecessary column
df.drop("skills_desc", axis=1, inplace=True)

# Posting effectiveness
df['posting_effectiveness'] = df['applies'].apply(lambda x: 'Bad' if x == 0 else ('Okay' if 0 < x < 10 else ('Effective' if x < 100 else 'Too Many')))
print(df['posting_effectiveness'])

# Word cloud for job titles
job_titles = df['title'].dropna()
all_titles_text = ' '.join(job_titles)
stopwords = set(STOPWORDS)
all_titles_text = ' '.join([word for word in all_titles_text.split() if word.lower() not in stopwords])
all_titles_text = re.sub(r'\W', ' ', all_titles_text)
words = all_titles_text.split()
word_counter = Counter(words)
most_common_words = word_counter.most_common(100)
print("100 Most Common Words in 'title' column:")
for word, count in most_common_words:
    print(f"{word}: {count}")

# Define the most common words
most_common_words_l = [word for word, _ in most_common_words]

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
