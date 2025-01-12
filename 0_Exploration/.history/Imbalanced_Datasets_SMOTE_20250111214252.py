I want to consider the difference with and without smote on the accuracy of the model

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Sample dataset
data = {
    'feature1': [1, 2, 3, 4, 5, 6, 7,8 , 9, 10],
    'feature2': [5, 4, 3, 2, 1, 0, 1, 2, 3, 4],
    'gender': ['male', 'male', 'male', 'male', 'male', 'female', 'female', 'female', 'female', 'female']
}

df = pd.DataFrame(data)

# Split data into features and target
X = df[['feature1', 'feature2']]
y = df['gender']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train a classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_resampled, y_resampled)

# Predict and evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))