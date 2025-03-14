# %% [markdown]
# <h2>Categorical Variables and One Hot Encoding</h2>

# %%
import pandas as pd

# %%
df = pd.read_csv("homeprices.csv")
df

# %% [markdown]
# <h2 style='color:purple'>Using pandas to create dummy variables</h2>

# %%
dummies = pd.get_dummies(df.town)
dummies

# %%
merged = pd.concat([df,dummies],axis='columns')
merged

# %%
final = merged.drop(['town'], axis='columns')
final

# %% [markdown]
# <h3 style='color:purple'>Dummy Variable Trap</h3>

# %% [markdown]
# When you can derive one variable from other variables, they are known to be multi-colinear. Here
# if you know values of california and georgia then you can easily infer value of new jersey state, i.e. 
# california=0 and georgia=0. There for these state variables are called to be multi-colinear. In this
# situation linear regression won't work as expected. Hence you need to drop one column. 

# %% [markdown]
# **NOTE: sklearn library takes care of dummy variable trap hence even if you don't drop one of the 
#     state columns it is going to work, however we should make a habit of taking care of dummy variable
#     trap ourselves just in case library that you are using is not handling this for you**

# %%
final = final.drop(['west windsor'], axis='columns')
final

# %%
X = final.drop('price', axis='columns')
X

# %%
y = final.price

# %%
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# %%
model.fit(X,y)

# %%
model.predict(X) # 2600 sqr ft home in new jersey

# %%
model.score(X,y)

# %%
model.predict([[3400,0,0]]) # 3400 sqr ft home in west windsor

# %%
model.predict([[2800,0,1]]) # 2800 sqr ft home in robbinsville

# %% [markdown]
# <h2 style='color:purple'>Using sklearn OneHotEncoder</h2>

# %% [markdown]
# First step is to use label encoder to convert town names into numbers

# %%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# %%
dfle = df
dfle.town = le.fit_transform(dfle.town)
dfle

# %%
X = dfle[['town','area']].values

# %%
X

# %%
y = dfle.price.values
y

# %% [markdown]
# Now use one hot encoder to create dummy variables for each of the town

# %%
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('town', OneHotEncoder(), [0])], remainder = 'passthrough')

# %%
X = ct.fit_transform(X)
X

# %%
X = X[:,1:]

# %%
X

# %%
model.fit(X,y)

# %%
model.predict([[0,1,3400]]) # 3400 sqr ft home in west windsor

# %%
model.predict([[1,0,2800]]) # 2800 sqr ft home in robbinsville

# %% [markdown]
# <h2 style='color:green'>Exercise</h2>

# %% [markdown]
# At the same level as this notebook on github, there is an Exercise folder that contains carprices.csv.
# This file has car sell prices for 3 different models. First plot data points on a scatter plot chart
# to see if linear regression model can be applied. If yes, then build a model that can answer
# following questions,
# 
# **1) Predict price of a mercedez benz that is 4 yr old with mileage 45000**
# 
# **2) Predict price of a BMW X5 that is 7 yr old with mileage 86000**
# 
# **3) Tell me the score (accuracy) of your model. (Hint: use LinearRegression().score())**


