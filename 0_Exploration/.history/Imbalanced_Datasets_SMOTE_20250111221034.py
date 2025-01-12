import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Generate a larger and more imbalanced dataset
X, y = make_classification(
    n_samples=5000,           # Larger dataset
    n_features=10,            # 10 features
    n_informative=5,          # 5 informative features
    n_redundant=2,            # 2 redundant features
    n_clusters_per_class=2,   # Overlapping feature space
    weights=[0.99, 0.01],     # Severe class imbalance
    flip_y=0,                 # No label noise for clarity
    random_state=42
)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Without Resampling
clf_no_resampling = LogisticRegression(random_state=42, max_iter=1000)
clf_no_resampling.fit(X_train, y_train)
y_pred_no_resampling = clf_no_resampling.predict(X_test)

print("Classification Report Without Resampling:")
print(classification_report(y_test, y_pred_no_resampling))

# With SMOTE Resampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

clf_with_resampling = LogisticRegression(random_state=42, max_iter=1000)
clf_with_resampling.fit(X_resampled, y_resampled)
y_pred_with_resampling = clf_with_resampling.predict(X_test)

print("\nClassification Report With SMOTE Resampling:")
print(classification_report(y_test, y_pred_with_resampling))
