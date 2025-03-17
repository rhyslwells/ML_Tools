import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load dataset
FILEPATH = '../../../../Datasets/Binary_predictors.csv'
data = pd.read_csv(FILEPATH)

# Encode categorical variables
data['Admitted'] = data['Admitted'].map({'Yes': 1, 'No': 0})
data['Gender'] = data['Gender'].map({'Female': 1, 'Male': 0})

# Split dataset into training and testing sets
train_data = data.sample(frac=0.8, random_state=2000)
test_data = data.drop(train_data.index)

# Define target variable and predictors
y_train = train_data['Admitted']
X_train = sm.add_constant(train_data[['SAT', 'Gender']])

y_test = test_data['Admitted']
X_test = sm.add_constant(test_data[['SAT', 'Gender']])

# Fit logistic regression model
model = sm.Logit(y_train, X_train)
results = model.fit()

# Display regression summary
print(results.summary())

# Exponentiate coefficients for interpretation
exp_coef = np.exp(results.params)
print("Exponentiated Coefficients:", exp_coef)

# Compute training accuracy
def compute_accuracy(actual, predicted):
    """
    Compute accuracy given actual labels and predicted probabilities.
    """
    bins = np.array([0, 0.5, 1])
    cm = np.histogram2d(actual, predicted, bins=bins)[0]
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    return cm, accuracy

train_predictions = results.predict()
cm_train, accuracy_train = compute_accuracy(y_train, train_predictions)
print(f'Training Accuracy: {accuracy_train:.2f}')

# Compute confusion matrix and accuracy for test data
test_predictions = results.predict(X_test)
cm_test, accuracy_test = compute_accuracy(y_test, test_predictions)

# Format and display confusion matrix
cm_df = pd.DataFrame(cm_test, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1'])
print("Confusion Matrix:")
print(cm_df)

# Display test accuracy and misclassification rate
print(f'Test Accuracy: {accuracy_test:.2f}')
print(f'Misclassification Rate: {1 - accuracy_test:.2f}')
