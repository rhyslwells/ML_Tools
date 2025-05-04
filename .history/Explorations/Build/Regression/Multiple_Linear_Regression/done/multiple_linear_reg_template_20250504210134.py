import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn
seaborn.set()
from sklearn.linear_model import LinearRegression

# Load the data from a .csv in the same folder
# raw_data = pd.read_csv('real_estate_price_size_year.csv')

raw_data = pd.read_csv('real_estate_price_size_year_view.csv')

raw_data.describe(include='all')

#encode cats:
data = raw_data.copy()
data['view'] = data['view'].map({'Sea view': 1, 'No sea view': 0})
data

# Following the regression equation, our dependent variable (y) is the GPA
y = data ['price']
# Similarly, our independent variable (x) is the SAT score
# x1 = data [['SAT','Rand 1,2,3']]
x1 = data[['size','year','view']]

# Preprocessing

# #scale inputs
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(x1)
# x_scaled = scaler.transform(x1)

# x_scaled

#with statsmodels
# Add a constant. Esentially, we are adding a new column (equal in lenght to x), which consists only of 1s
x = sm.add_constant(x1)
# Fit the model, according to the OLS (ordinary least squares) method with a dependent variable y and an idependent x
results = sm.OLS(y,x).fit()
# Print a nice summary of the regression.
results.summary()

x.head()
new_data = sm.add_constant([[1,750, 2009, 0]])
prediction_for_new_data = results.predict(new_data)
print(prediction_for_new_data) #[231727.85029188]

#model with sklearn
#Remember with statsmodels there is an extra constant with x1
reg = LinearRegression()
reg.fit(x,y) # remember input can be scaled

# -  Display the intercept and coefficient(s)
reg.coef_
reg.intercept_

# -  Find the R-squared and Adjusted R-squared
reg.score(x,y)
# Let's use the handy function we created
def adj_r2(x,y):
    r2 = reg.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2
adj_r2(x1,y)

# -  Compare the R-squared and the Adjusted R-squared

# -  Using the model make a prediction 
new_data = [[750,2009,0]]
new_data_scaled = scaler.transform(new_data) #because we previously scaled x
reg.predict(new_data_scaled)
reg.predict(new_data) #array([231727.85029188])


# -  Find the univariate (or multivariate if you wish - see the article) p-values of the two variables. What can you say about them?

# -  Create a summary table with your findings

from sklearn.feature_selection import f_regression
p_values = f_regression(x,y)[1]
p_values

reg_summary = pd.DataFrame(data = x.columns.values, columns=['Features'])
reg_summary ['Coefficients'] = reg.coef_
reg_summary ['p-values'] = p_values.round(3)
reg_summary

# 	Features	Coefficients	p-values
# 0	const	0.000000	1.000
# 1	size	223.031619	0.000
# 2	year	2718.948889	0.357
# 3	view	56726.019798	0.000



# Model evaluation

statsmodel summary see adjusted squared

See the effects of creating dummies a summary before and after

data = raw_data.copy()
data['view'] = data['view'].map({'Sea view': 1, 'No sea view': 0})