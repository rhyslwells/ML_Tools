# Import necessary libraries
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# Define text preprocessing function
def text_to_words(text):
    text = BeautifulSoup(text, 'lxml').get_text()  # Clean HTML
    letters = re.sub('[^a-zA-Z]', ' ', text)      # Remove non-alphabetic characters
    words = letters.lower().split()               # Split into words
    stops = set(stopwords.words('english'))       # Get stopwords
    meaningful_words = [w for w in words if not w in stops]  # Remove stopwords
    return (' '.join(meaningful_words))

# Generalized function to clean a dataset
def clean_data(df):
    return [text_to_words(review) for review in df]

# Example dataset (replace this with any dataset)
data = pd.DataFrame({
    'review': ["This is a great movie", "I hated the movie", "Not bad at all", "It was amazing!"]
})

# Convert text data into words
cleaned_reviews = clean_data(data['review'])

# Apply Bag of Words
vectorizer = CountVectorizer(max_df=0.5, max_features=10000)
data_features = vectorizer.fit_transform(cleaned_reviews)

# Analyze the transformed data
feature_names = vectorizer.get_feature_names_out()
print("Sample of features: ", feature_names[:10])  # Print first 10 words
