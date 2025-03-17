import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Apply a fix to the statsmodels library
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

# Load the dataset
raw_data = pd.read_csv('../../../../Datasets/Binary_predictors.csv')
data = raw_data.copy()
data['Admitted'] = data['Admitted'].map({'Yes': 1, 'No': 0})
data['Gender'] = data['Gender'].map({'Female': 1, 'Male': 0})

data.head()

# Declare the dependent and independent variables
y = data['Admitted']
x1 = data[['SAT', 'Gender']]

# Logistic Regression
x = sm.add_constant(x1)
reg_log = sm.Logit(y, x)
results_log = reg_log.fit()

# Regression summary
results_log.summary()

# Exponentiate coefficient
np.exp(1.94)

# Accuracy Calculation
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
results_log.predict()

# Convert Admitted column to NumPy array
np.array(data['Admitted'])

# Confusion matrix
cm_df = pd.DataFrame(results_log.pred_table())
cm_df.columns = ['Predicted 0', 'Predicted 1']
cm_df = cm_df.rename(index={0: 'Actual 0', 1: 'Actual 1'})
cm_df

# Compute training accuracy
cm = np.array(cm_df)
accuracy_train = (cm[0, 0] + cm[1, 1]) / cm.sum()
accuracy_train

# Load and preprocess the test dataset
test = data.copy()

test['Admitted'] = test['Admitted'].map({'Yes': 1, 'No': 0})
test['Gender'] = test['Gender'].map({'Female': 1, 'Male': 0})

test_actual = test['Admitted']
test_data = test.drop(['Admitted'], axis=1)
test_data = sm.add_constant(test_data)

# Why is this needed
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
    bins = np.array([0, 0.5, 1])
    cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    return cm, accuracy

# Compute confusion matrix for test data
cm = confusion_matrix(test_data, test_actual, results_log)

# Format confusion matrix as DataFrame
cm_df = pd.DataFrame(cm[0])
cm_df.columns = ['Predicted 0', 'Predicted 1']
cm_df = cm_df.rename(index={0: 'Actual 0', 1: 'Actual 1'})
cm_df

# Compute misclassification rate
missclassification_rate = (1 + 1) / 19
print(f'Misclassification rate: {missclassification_rate:.2f}')
