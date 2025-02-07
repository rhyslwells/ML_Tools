# Use the example of spam prediction. 

# Exploring precision metric

from sklearn.metrics import precision_score

# Sample true labels and predicted labels
y_true = [0, 1, 1, 0, 1, 0, 1, 1]
y_pred = [0, 1, 0, 0, 1, 1, 1, 1]

# Calculate precision score
precision = precision_score(y_true, y_pred)

print("Precision Score: {:.2f}".format(precision))

# Exploring the classification report

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Assuming X and y are your features and labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Model 1
model_1 = RandomForestClassifier(n_estimators=2)
model_1.fit(X_train, y_train)
y_predicted_1 = model_1.predict(X_test)
print("Classification Report for Model 1:")
print(classification_report(y_test, y_predicted_1))

# Model 2
model_2 = RandomForestClassifier(n_estimators=10)
model_2.fit(X_train, y_train)
y_predicted_2 = model_2.predict(X_test)
print("\nClassification Report for Model 2:")
print(classification_report(y_test, y_predicted_2))

# Recal 

from sklearn.metrics import recall_score

# Sample true labels and predicted labels
y_true = [0, 1, 1, 0, 1, 0, 1, 1]
y_pred = [0, 1, 0, 0, 1, 1, 1, 1]

# Calculate recall score
recall = recall_score(y_true, y_pred)

print("Recall Score: {:.2f}".format(recall))