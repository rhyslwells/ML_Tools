# %% [markdown]
# <a href="https://colab.research.google.com/github/MatthewK84/LinkedIn-Learning-Journey/blob/main/ARXIV_Time_Series_Final.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
!pip install arxiv --user

# %%
import pandas as pd
import arxiv

# %%
# i only use these if I want to remove annoying deprecation warnings from my analysis
import warnings
warnings.filterwarnings('ignore')

# %%
def search_arxiv(query, max_results=10):

    data = {}
    i = 0

    search = arxiv.Search(query=query, max_results=max_results)

    for result in search.results():

        try:

            data[i] = {}

            data[i]['title'] = result.title
            data[i]['date_published'] = result.published
            data[i]['authors'] = [a.name for a in result.authors]
            data[i]['summary'] = result.summary
            data[i]['url'] = result.pdf_url
            data[i]['category'] = result.primary_category

        except:

            print('weird arxiv error')

        # there are more fields that can be added; add as many as you need

        i += 1

    df = pd.DataFrame(data).T
    df = df[['date_published', 'title', 'authors', 'summary', 'url', 'category']]
    df['date_published'] = pd.to_datetime(df['date_published'])
    df.sort_values('date_published', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# %%
# at 1000 articles, it takes about 30 seconds
# at 10000 articles, it takes a while; put laptop down and walk away
# i haven't tried beyond 10000 yet

# at the moment, this is finicky beyond 10000; needs debugged or something
# could also be a problem with the arxiv library as it seems to be inside the search

# so far, i haven't noticed obvious throttling, but possibly throttling is causing the issue

query = 'Time Series'

max_results = 5100

df = search_arxiv(query, max_results)
df.head()

# %%
df.shape

# %%
outfile = '/content/sample_data/arxix_time_series_data.csv'

df.to_csv(outfile, index=False)

# %%
Time_Series = pd.read_csv('/content/sample_data/arxix_time_series_data.csv')
Time_Series.head()

# %%
Time_Series['category'].value_counts()

# %%
import csv

# The path to the CSV file
input_file_path = '/content/sample_data/arxix_time_series_data.csv'
output_file_path = '/content/sample_data/time_series_filtered.csv'


# List of categories to include
included_categories = ['cs.LG', 'stat.ME', 'math.ST', 'stat.ML']

# Read the CSV file and filter rows
with open(input_file_path, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    fieldnames = reader.fieldnames  # Capture the fieldnames for writing

    # Write the filtered rows to a new CSV file
    with open(output_file_path, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()  # Write the header to the output file

        for row in reader:
            # Check if the 'category' is in the list of included categories
            if row['category'] in included_categories:
                writer.writerow(row)

# %%
Time_Series_Final = pd.read_csv('/content/sample_data/time_series_filtered.csv')
Time_Series_Final.head()

# %%
Time_Series_Final.shape

# %%
import csv
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# The path to the CSV file
input_file_path = '/content/sample_data/time_series_filtered.csv'

# Common English stop words
stop_words = set(['as','is','a','are','it','paper','based','proposed', 'an','approach', 'using', 'time-series', 'learning', 'performance', 'results', 'these', 'show', 'have', 'has','datasets','propose','classification','two','also','used','different','analysis','problem','new','framework', 'novel','been','demonstrate',
                  'that', 'this', 'be', 'which', 'proposed', 'time', 'data', 'method', 'methods', 'our', 'we', 'series', 'forecasting', 'model', 'models', 'the', 'model', 'and', 'or', 'but', 'if', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'many', 'however', 'existing', 'its', 'prediction', 'application',
                  'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])

# Initialize a Counter for word frequencies
word_freq = Counter()

# Read the CSV file and process the 'summary' column
with open(input_file_path, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)

    for row in reader:
        summary = row['summary']
        # Tokenize the summary text
        words = summary.split()
        # Filter out stop words and count word frequencies
        for word in words:
            word = word.lower().rstrip('.!,')
            if word not in stop_words:
                word_freq[word] += 1

# Select the ten most common words
most_common_words = word_freq.most_common(10)

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color ='white').generate_from_frequencies(dict(most_common_words))

# Display the word cloud using matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Remove the axis
plt.show()

# %%
import pandas as pd

# Sample lists of Foundation and Advanced Methods
Foundation_keywords = ['ARIMA', 'SAMIRA', 'LSTM', 'Conformal Prediction', 'multivariate time series', 'vector autoregression']
Advanced_Methods_keywords = ['Transformer', 'GAN', 'EDAIN', 'MLP', 'deep learning', 'neural networks', 'supervised learning']

# Function to analyze sentiment based on word lists
def simple_sentiment_analysis(text):
    # Tokenize the text by words and convert to lower case
    words = text.lower().split()
    # Count the number of Foundation and Advanced Methods words in the text
    foundation_count = sum(words.count(word) == 1 for word in Foundation_keywords)
    advanced_methods_count = sum(words.count(word) == 1 for word in Advanced_Methods_keywords)
    # Determine the sentiment based on the counts
    if foundation_count >= advanced_methods_count:
        return 'Foundation'
    elif advanced_methods_count <= foundation_count:
        return 'Advanced Methods'
    else:
        return 'Mixed'

# Load the sample CSV file into a DataFrame
df = pd.read_csv('/content/sample_data/time_series_filtered.csv')

# Apply the simple sentiment analysis to each summary
df['sentiment_interpretation'] = df['summary'].apply(simple_sentiment_analysis)

# Show the DataFrame with the sentiment analysis
df[['date_published', 'summary', 'sentiment_interpretation']]

# %%
import pandas as pd
import re

# Sample lists of Foundation and Advanced Methods in lowercase for case-insensitive comparison
Foundation_keywords = ['arima', 'samira', 'lstm', 'conformal prediction', 'multivariate time series', 'vector autoregression']
Advanced_Methods_keywords = ['transformer', 'gan', 'edain', 'mlp', 'deep learning', 'neural networks', 'supervised learning']

# Function to count occurrences of keywords based on word lists
def count_keyword_occurrences(text):
    # Use regular expressions to split the text into words and convert to lower case
    words = re.findall(r'\b\w+\b', text.lower())
    # Count the number of distinct Foundation and Advanced Methods words in the text
    foundation_count = sum(word in Foundation_keywords for word in words)
    advanced_methods_count = sum(word in Advanced_Methods_keywords for word in words)
    return foundation_count, advanced_methods_count

# Load the sample CSV file into a DataFrame
df = pd.read_csv('/content/sample_data/time_series_filtered.csv')

# Apply the count_keyword_occurrences function to each summary and split the results into two columns
df[['foundation_count', 'advanced_methods_count']] = df.apply(
    lambda row: pd.Series(count_keyword_occurrences(row['summary'])),
    axis=1
)

# Ensure the 'date_published' column is present or adjust the column name as necessary
# Show the DataFrame with the counts
df[['date_published', 'summary', 'foundation_count', 'advanced_methods_count']]

# %%
# Count the occurrences of each 'foundation_count' value excluding zeros
foundation_count_values = df['foundation_count'].value_counts().sort_index()
foundation_count_values = foundation_count_values[foundation_count_values.index != 0]

# Create a bar chart without the zero count
plt.figure(figsize=(10, 6))
foundation_count_values.plot(kind='bar', color='skyblue')
plt.title('Foundation Count Frequency (Excluding Zero Count)')
plt.xlabel('Foundation Count')
plt.ylabel('Frequency')
plt.xticks(rotation=0)  # To keep the x-axis labels horizontal
plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
plt.show()

# %%
# Count the occurrences of each 'advanced_methods_count' value excluding zeros
advanced_methods_count_values = df['advanced_methods_count'].value_counts().sort_index()
advanced_methods_count_values = advanced_methods_count_values[advanced_methods_count_values.index != 0]

# Create a bar chart without the zero count for advanced methods
plt.figure(figsize=(10, 6))
advanced_methods_count_values.plot(kind='bar', color='coral')
plt.title('Advanced Methods Count Frequency (Excluding Zero Count)')
plt.xlabel('Advanced Methods Count')
plt.ylabel('Frequency')
plt.xticks(rotation=0)  # To keep the x-axis labels horizontal
plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
plt.show()


