import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils import resample

# Sample dataset
data = {
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature2': [5, 4, 3, 2, 1, 0, 1, 2, 3, 4],
    'gender': ['male', 'male', 'male', 'male', 'male', 'female', 'female', 'female', 'female', 'female']
}

df = pd.DataFrame(data)

# Split data into features and target
X = df[['feature1', 'feature2']]
y = df['gender']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scenario 1: Train without resampling
clf_no_resampling = RandomForestClassifier(random_state=42)
clf_no_resampling.fit(X_train, y_train)
y_pred_no_resampling = clf_no_resampling.predict(X_test)
print("Classification Report Without Resampling:")
print(classification_report(y_test, y_pred_no_resampling))

# Scenario 2: Train with random oversampling
# Combine features and target for resampling
train_data = pd.concat([X_train, y_train], axis=1)

# Separate majority and minority classes
majority_class = train_data[train_data['gender'] == 'male']
minority_class = train_data[train_data['gender'] == 'female']

# Perform random oversampling
minority_oversampled = resample(
    minority_class,
    replace=True,  # Sample with replacement
    n_samples=len(majority_class),  # Match majority class size
    random_state=42
)

# Combine the oversampled minority class with the majority class
balanced_data = pd.concat([majority_class, minority_oversampled])

# Separate features and target after resampling
X_resampled = balanced_data[['feature1', 'feature2']]
y_resampled = balanced_data['gender']

# Train a classifier on the resampled data
clf_with_resampling = RandomForestClassifier(random_state=42)
clf_with_resampling.fit(X_resampled, y_resampled)

# Predict and evaluate
y_pred_with_resampling = clf_with_resampling.predict(X_test)
print("\nClassification Report With Random Oversampling:")
print(classification_report(y_test, y_pred_with_resampling))
