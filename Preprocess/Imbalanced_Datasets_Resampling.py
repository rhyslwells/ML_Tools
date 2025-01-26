import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils import resample

# Generate a larger, imbalanced synthetic dataset
X, y = make_classification(
    n_samples=1000,  # Total samples
    n_features=5,    # Number of features
    n_informative=3, # Number of informative features
    n_redundant=0,   # No redundant features
    n_clusters_per_class=1, 
    weights=[0.95, 0.05],  # Imbalance: 80% majority, 20% minority
    random_state=42
)

# Convert to DataFrame for easier handling
df = pd.DataFrame(X, columns=[f'feature{i+1}' for i in range(X.shape[1])])
df['target'] = y

# Split data into features and target
X = df.drop('target', axis=1)
y = df['target']


# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scenario 1: Without Resampling
clf_no_resampling = RandomForestClassifier(random_state=42)
clf_no_resampling.fit(X_train, y_train)
y_pred_no_resampling = clf_no_resampling.predict(X_test)
print("Classification Report Without Resampling:")
print(classification_report(y_test, y_pred_no_resampling))

# Scenario 2: With Random Oversampling
# Combine features and target for resampling
train_data = pd.concat([X_train, y_train], axis=1)

# Separate majority and minority classes
majority_class = train_data[train_data['target'] == 0]
minority_class = train_data[train_data['target'] == 1]

# Perform random oversampling
minority_oversampled = resample(
    minority_class,
    replace=True,  # Sample with replacement
    n_samples=len(majority_class),  # Match majority class size
    random_state=42
)

# Combine oversampled minority class with majority class
balanced_data = pd.concat([majority_class, minority_oversampled])

# Separate features and target after resampling
X_resampled = balanced_data.drop('target', axis=1)
y_resampled = balanced_data['target']

# Train a classifier on the resampled data
clf_with_resampling = RandomForestClassifier(random_state=42)
clf_with_resampling.fit(X_resampled, y_resampled)

# Predict and evaluate
y_pred_with_resampling = clf_with_resampling.predict(X_test)
print("\nClassification Report With Random Oversampling:")
print(classification_report(y_test, y_pred_with_resampling))
