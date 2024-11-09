#imports

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

#load data
raw_data = pd.read_csv('file.csv')
raw_data

#preprocessing

# We make sure to create a copy of the data before we start altering it. Note that we don't change the original data we loaded.
data = raw_data.copy()
# Removes the index column thata comes with the data
data = data.drop(["var1"], axis = 1)

#encode categroicals
data['var2'] = data['var2'].map({'yes':1, 'no':0})
data

# model setup
y = data['y'] #target 
x1 = data['duration'] #independant var
# x1= data[['duration',"credit"]] # can have mulple

#sm logregression

x = sm.add_constant(x1)
reg_log = sm.Logit(y,x)
results_log = reg_log.fit()
results_log.summary()
coefficients = results_log.params
coefficients

#
"""
The Pseudo R-squared is 0.21 which is within the 'acceptable region'. 
"""

# plot

# Create a scatter plot of x1 (Duration, no constant) and y (Subscribed)
plt.scatter(x1,y,color = 'C0')
# Don't forget to label your axes!
plt.xlabel('x1', fontsize = 20)
plt.ylabel('y', fontsize = 20)
plt.show()

# Log odds: the odds of duration are the exponential of the log odds from the summary table
np.exp(0.0051)