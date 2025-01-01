import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Sample dataset
data = {
    'feature1': [1, 2, 3, 4, 5, 6, 7, , 9, 10],
    'feature2': [5, 4, 3, 2, 1, 0, 1, 2, 3, 4],
    'gender': ['male', 'male', 'male', 'male', 'male', 'female', 'female', 'female', 'female', 'female']
}

df = pd.DataFrame(data)

# Split data into features and target
X = df[['feature1', 'feature2']]
y = df['gender']