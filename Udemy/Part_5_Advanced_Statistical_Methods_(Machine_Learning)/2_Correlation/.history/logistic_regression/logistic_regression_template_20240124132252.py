# Imports
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Load data
raw_data = pd.read_csv('Bank_data.csv')
raw_data

# Preprocessing

# Create a copy of the data before altering it. Do not change the original loaded data.
data = raw_data.copy()

# Remove the index column that comes with the data
data = data.drop(["var1"], axis=1)

# Encode categoricals
data['var2'] = data['var2'].map({'yes': 1, 'no': 0})
data

# Model setup
y = data['y']  # Target / dependent
estimators = ['interest_rate', 'march', 'credit', 'previous', 'duration']
x1 = data[estimators]
# x1 = data[['duration', 'credit']]  # Can have multiple independent variables

# Statsmodels logistic regression

x = sm.add_constant(x1)
reg_log = sm.Logit(y, x)
results_log = reg_log.fit()
print(results_log.summary())
coefficients = results_log.params
coefficients

# Pseudo R-squared interpretation
"""
The Pseudo R-squared is 0.21, which falls within the 'acceptable region'.
"""

# Plotting

# Creating a logit function for plotting the curve
def f(x, b0, b1):
    return np.array(np.exp(b0 + x * b1) / (1 + np.exp(b0 + x * b1)))

# Sorting the y and x for plotting the curve
f_sorted = np.sort(f(x1, results_log.params[0], results_log.params[1]))
x_sorted = np.sort(np.array(x1))
plt.scatter(x1, y, color='C0')
# plt.xlabel('SAT', fontsize=20)
# plt.ylabel('Admitted', fontsize=20)
plt.plot(x_sorted, f_sorted, color='red')
plt.figure(figsize=(20, 20))
plt.xlabel('x1', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.show()

# Log odds interpretation
# The odds of duration are the exponential of the log odds from the coefficient
np.exp(0.0051)


# Create an evaluation matrix ect

# Want to do evaluation of model

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# model setup
y = data['y']  # target / dependent
estimators = ['interest_rate', 'march', 'credit', 'previous', 'duration']
X = data[estimators]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# sklearn logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Print the coefficients
coefficients = logreg.coef_
intercept = logreg.intercept_
print(f'Coefficients: {coefficients}')
print(f'Intercept: {intercept}')

from sklearn.metrics import classification_report, confusion_matrix

y_pred = logreg.predict(X_test)

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

