# imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()

# load data
path='1.01. Simple linear regression.csv'
df = pd.read_csv(path)

y = df ['var2']
x1 = df ['var1']

# scatter plot
plt.scatter(x1,y)
plt.xlabel('var1', fontsize = 20)
plt.ylabel('var2', fontsize = 20)
plt.show()

# Regression

# Add a constant. Essentially, we are adding a new column (equal in lenght to x), which consists only of 1s
x = sm.add_constant(x1)
# Fit the model, according to the OLS (ordinary least squares) method with a dependent variable y and an idependent x
results = sm.OLS(y,x).fit()
results.summary()

# Create a scatter plot
plt.scatter(x1,y)
yhat = 0.0017*x1 + 0.275 # Picked from table above

# Plot the regression line against the independent variable (SAT)
fig = plt.plot(x1,yhat, lw=4, c='orange', label ='regression line')
# Label the axes
plt.xlabel('var1', fontsize = 20)
plt.ylabel('var2', fontsize = 20)
plt.show()