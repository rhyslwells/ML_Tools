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
y = data['y'] #target /dependnat
# x1 = data['duration'] #independant var
estimators=['interest_rate','march','credit','previous','duration']
x1 = data[estimators]
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
# Creating a logit regression (we will discuss this in another notebook)
# Creating a logit function, depending on the input and coefficients
def f(x,b0,b1):
    return np.array(np.exp(b0+x*b1) / (1 + np.exp(b0+x*b1)))

# Sorting the y and x, so we can plot the curve
f_sorted = np.sort(f(x1,results_log.params[0],results_log.params[1]))
x_sorted = np.sort(np.array(x1))
ax = plt.scatter(x1,y,color='C0')
#plt.xlabel('SAT', fontsize = 20)
#plt.ylabel('Admitted', fontsize = 20)
# Plotting the curve
ax2 = plt.plot(x_sorted,f_sorted,color='red')
plt.figure(figsize=(20,20))
plt.xlabel('x1', fontsize = 20)
plt.ylabel('y', fontsize = 20)
plt.show()

# Log odds: the odds of duration are the exponential of the log odds from the summary table
# from coefficents
np.exp(0.0051)

# Create an evaluation matrix ect