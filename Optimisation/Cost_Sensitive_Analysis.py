from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve

# Create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.7, 0.3], random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the cost matrix
cost_matrix = {
    'C(TP)': 0,  # True Positive Cost
    'C(FP)': 1,  # False Positive Cost
    'C(TN)': 0,  # True Negative Cost
    'C(FN)': 5   # False Negative Cost
}

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)



# Train the classifier
clf.fit(X_train, y_train)

# Get the predicted probabilities
y_probs = clf.predict_proba(X_test)[:, 1]

# Calculate precision-recall curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

# Find the threshold that minimizes the total cost
min_cost = float('inf')
best_threshold = 0.5
for threshold in thresholds:
    y_pred = (y_probs >= threshold).astype(int)
    conf_matrix = confusion_matrix(y_test, y_pred)
    total_cost = (conf_matrix[0, 1] * cost_matrix['C(FP)'] + 
                  conf_matrix[1, 0] * cost_matrix['C(FN)'])
    if total_cost < min_cost:
        min_cost = total_cost
        best_threshold = threshold

print(f"Best Threshold: {best_threshold}")
print(f"Minimum Total Cost: {min_cost}")

# Use the best threshold to make final predictions
y_pred_final = (y_probs >= best_threshold).astype(int)