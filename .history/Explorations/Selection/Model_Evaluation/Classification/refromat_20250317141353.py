import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load dataset
filepath = '../../../../Datasets/Binary_predictors.csv'
data = pd.read_csv(filepath)
data['Admitted'] = data['Admitted'].map({'Yes': 1, 'No': 0})
data['Gender'] = data['Gender'].map({'Female': 1, 'Male': 0})

#Set aside some data for testing
train_data = data.sample(frac=0.8, random_state=2000)
test_data = data.drop(train_data.index)


y = train_data['Admitted']  # Define target variable
x1 = train_data[['SAT', 'Gender']]  # Define independent variables

# Add a constant to the independent variables matrix
x = sm.add_constant(x1)

# Fit logistic regression model
reg_log = sm.Logit(y, x)
results_log = reg_log.fit()

# Display regression summary
print(results_log.summary())

# Exponentiate coefficient for interpretation
exp_coef = np.exp(results_log.params)
print("Exponentiated Coefficients:", exp_coef)

# Compute training accuracy
train_predictions = results_log.predict()
bins = np.array([0, 0.5, 1])
cm_train = np.histogram2d(y, train_predictions, bins=bins)[0]
accuracy_train = (cm_train[0, 0] + cm_train[1, 1]) / cm_train.sum()
print(f'Training Accuracy: {accuracy_train:.2f}')

# Function to compute confusion matrix and accuracy
def confusion_matrix(data, actual_values, model):
    """
    Generate a confusion matrix and compute accuracy.
    
    Parameters:
    data : DataFrame
        Test dataset formatted like training data (excluding actual values).
    actual_values : array-like
        True labels (0s and 1s).
    model : LogitResults
        Fitted logistic regression model.
    
    Returns:
    tuple
        Confusion matrix and accuracy.
    """
    pred_values = model.predict(data)
    cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    return cm, accuracy

# Load test dataset (use train_data if test data is unavailable)



test_actual = test_data['Admitted']
test_features = test_data.drop(['Admitted'], axis=1)
test_features = sm.add_constant(test_features)

# Compute confusion matrix for test data
cm, accuracy_test = confusion_matrix(test_features, test_actual, results_log)
print(f'Test Accuracy: {accuracy_test:.2f}')

# Format confusion matrix as DataFrame
cm_df = pd.DataFrame(cm, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1'])
print("Confusion Matrix:")
print(cm_df)

# Compute misclassification rate
misclassification_rate = 1 - accuracy_test
print(f'Misclassification Rate: {misclassification_rate:.2f}')