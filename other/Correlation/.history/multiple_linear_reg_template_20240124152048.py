import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn
seaborn.set()
from sklearn.linear_model import LinearRegression

# Load the data from a .csv in the same folder
raw_data = pd.read_csv('real_estate_price_size_year.csv')


raw_data.describe(include='all')

#encode cats:
data = raw_data.copy()
data['view'] = data['view'].map({'Sea view': 1, 'No sea view': 0})

# Following the regression equation, our dependent variable (y) is the GPA
y = data ['GPA']
# Similarly, our independent variable (x) is the SAT score
# x1 = data [['SAT','Rand 1,2,3']]
x1 = data[['size','year','view']]

# Preprocessing

#scale inputs
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x1)
x_scaled = scaler.transform(x1)



# Add a constant. Esentially, we are adding a new column (equal in lenght to x), which consists only of 1s
x = sm.add_constant(x1)
# Fit the model, according to the OLS (ordinary least squares) method with a dependent variable y and an idependent x
results = sm.OLS(y,x).fit()
# Print a nice summary of the regression.
results.summary()


#model with sklearn
reg = LinearRegression()
reg.fit(x_scaled,y)


# -  Display the intercept and coefficient(s)
reg.coef_
reg.intercept_

# -  Find the R-squared and Adjusted R-squared
reg.score(x_scaled,y)
# Let's use the handy function we created
def adj_r2(x,y):
    r2 = reg.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2
adj_r2(x_scaled,y)

# -  Compare the R-squared and the Adjusted R-squared

# -  Using the model make a prediction 
new_data = [[750,2009]]
new_data_scaled = scaler.transform(new_data) #because we previously scaled x
reg.predict(new_data_scaled)

# -  Find the univariate (or multivariate if you wish - see the article) p-values of the two variables. What can you say about them?

# -  Create a summary table with your findings

from sklearn.feature_selection import f_regression
p_values = f_regression(x,y)[1]
p_values

reg_summary = pd.DataFrame(data = x.columns.values, columns=['Features'])
reg_summary ['Coefficients'] = reg.coef_
reg_summary ['p-values'] = p_values.round(3)
reg_summary