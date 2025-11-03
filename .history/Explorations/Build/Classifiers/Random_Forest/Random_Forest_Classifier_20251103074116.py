'''
Use famous iris flower dataset from sklearn.datasets to predict flower 
species using random forest classifier.
1. Measure prediction score using default n_estimators (10)
2. Now fine tune your model by changing number of trees in your classifer and tell me
 what best score you can get using how many trees


 Results change dependnat on the random state, why?
'''
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
    from sklearn.metrics import r2_score
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report


# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
dir(iris) # sklearn data sets
iris.keys()
iris.target_names # labeling of certain plants
iris.target
df.head()
df['target'] = iris.target



# Define features and target
X = df.drop('target', axis='columns')
y = df.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Default n_estimators = 10
model_1 = RandomForestClassifier(n_estimators=10, random_state=47)
model_1.fit(X_train, y_train)
y_pred_1 = model_1.predict(X_test)

# evaluation:
y_predicted = model_1.predict(X_test)

#What can we seen from the classification report?
print(classification_report(y_test, y_predicted))

# Cross-validation for default model
cv_scores_1 = cross_val_score(model_1, X, y, cv=5)
print(f"Mean cross-validation accuracy for Model 1: {cv_scores_1.mean():.2f}")


cv_num=5
rng_num=0
# Fine-Tuning n_estimators with Cross-Validation
n_estimators_range = range(2, 51, 2)
cv_means = []
test_scores = []

for n in n_estimators_range:
    model = RandomForestClassifier(n_estimators=n, random_state=np.random.RandomState(0)) # want CV to be robust wrt estimator as  allow the estimator RNG to vary for each fold.
    
    # Cross-validation accuracy
    cv_scores = cross_val_score(model, X, y, cv=cv_num)
    cv_mean = cv_scores.mean()
    cv_means.append(cv_mean)
    
    # Train-test split accuracy
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)
    test_scores.append(test_score)

# Identify best number of trees based on cross-validation
best_cv_score = max(cv_means)
best_n_estimators_cv = n_estimators_range[cv_means.index(best_cv_score)]
print(f"\nBest cross-validation accuracy: {best_cv_score:.2f} achieved with n_estimators={best_n_estimators_cv}")

# Plot accuracy vs n_estimators
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, cv_means, marker='o', label='Cross-Validation Accuracy')
plt.plot(n_estimators_range, test_scores, marker='x', label='Test Set Accuracy')
plt.title("Accuracy vs Number of Trees (n_estimators)")
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

#------------------ Range of hyperparamaters ------------------

# for the above I want to plot the value for rng_num=0,1,2 and cv_num=3,4,5

# Define parameters for exploration
n_estimators_range = range(2, 51, 2)
rng_nums = [0]
cv_nums = [5,10]

# Initialize results dictionary
results = {}

# Loop over rng_num and cv_num
for rng_num in rng_nums:
    for cv_num in cv_nums:
        cv_means = []
        test_scores = []
        
        # Loop over n_estimators
        for n in n_estimators_range:
            # Use a varying RNG for cross-validation to test robustness
            rng = np.random.RandomState(rng_num)
            model = RandomForestClassifier(n_estimators=n, random_state=rng)
            
            # Cross-validation accuracy
            cv_scores = cross_val_score(model, X, y, cv=cv_num)
            cv_means.append(cv_scores.mean())
            
            # Train-test split accuracy
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            test_scores.append(accuracy_score(y_test, y_pred))
        
        # Store results
        results[(rng_num, cv_num)] = {
            'cv_means': cv_means,
            'test_scores': test_scores
        }

# Plot results
plt.figure(figsize=(12, 8))
for (rng_num, cv_num), metrics in results.items():
    label = f"RNG={rng_num}, CV={cv_num}"
    plt.plot(n_estimators_range, metrics['cv_means'], marker='o', label=f"{label} (CV Mean)")
    plt.plot(n_estimators_range, metrics['test_scores'], marker='x', linestyle='--', label=f"{label} (Test Score)")

plt.title("Accuracy vs Number of Trees (n_estimators) for Various RNG and CV Settings")
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

#---------------------------------------------------------------------

# Suppose we want to compare classifcation reports for two separate models on the 
# same test data but with more estimators for the second model.

# model 1
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=3)
model_1 = RandomForestClassifier(n_estimators=2)
model_1.fit(X_train, y_train)
y_predicted_1 = model_1.predict(X_test)
print("Classification Report for Model 1:")
print(classification_report(y_test, y_predicted_1))

# Model 2
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=3)
model_2 = RandomForestClassifier(n_estimators=10)
model_2.fit(X_train, y_train)
y_predicted_2 = model_2.predict(X_test)
print("\nClassification Report for Model 2:")
print(classification_report(y_test, y_predicted_2))

#--------------------------------------------------------------------------------

#Q: Is Model 2 overfit? 

from sklearn.model_selection import cross_val_score

# Perform cross-validation
cv_scores = cross_val_score(model_2, X, y, cv=5)  # Use 5-fold cross-validation

# Print cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())


#--------------------------------------------------------------------------------

#Q: What happens to the cross validation if we increase the number of estimators?

from sklearn.metrics import accuracy_score

# Model 3
model_3 = RandomForestClassifier(n_estimators=20, random_state=3)  # Increase the number of estimators
model_3.fit(X_train, y_train)

# Training set performance
y_train_pred_3 = model_3.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred_3)
print("Training Accuracy for Model 3:", train_accuracy)

# Test set performance
y_test_pred_3 = model_3.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred_3)
print("Test Accuracy for Model 3:", test_accuracy)

# Perform cross-validation for Model 3
cv_scores_3 = cross_val_score(model_3, X, y, cv=5)  # Use 5-fold cross-validation

# Print cross-validation scores for Model 3
print("Cross-Validation Scores for Model 3:", cv_scores_3)
print("Mean CV Score for Model 3:", cv_scores_3.mean())
