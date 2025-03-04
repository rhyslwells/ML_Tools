# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Suppress Warnings for clean notebook
import warnings
warnings.filterwarnings('ignore')
**We are going to use Melbourne House Price Dataset where we'll predict House Predictions based on various features.**
#### The Dataset Link is
https://www.kaggle.com/anthonypino/melbourne-housing-market
# read dataset
dataset = pd.read_csv('../Data/Melbourne_housing_FULL.csv')
dataset.head()
dataset.nunique()
# let's use limited columns which makes more sense for serving our purpose
cols_to_use = ['Suburb', 'Rooms', 'Type', 'Method', 'SellerG', 'Regionname', 'Propertycount', 
               'Distance', 'CouncilArea', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Price']
dataset = dataset[cols_to_use]
dataset.head()
dataset.shape
#### Checking for Nan values
dataset.isna().sum()
#### Handling Missing values
# Some feature's missing values can be treated as zero (another class for NA values or absence of that feature)
# like 0 for Propertycount, Bedroom2 will refer to other class of NA values
# like 0 for Car feature will mean that there's no car parking feature with house
cols_to_fill_zero = ['Propertycount', 'Distance', 'Bedroom2', 'Bathroom', 'Car']
dataset[cols_to_fill_zero] = dataset[cols_to_fill_zero].fillna(0)

# other continuous features can be imputed with mean for faster results since our focus is on Reducing overfitting
# using Lasso and Ridge Regression
dataset['Landsize'] = dataset['Landsize'].fillna(dataset.Landsize.mean())
dataset['BuildingArea'] = dataset['BuildingArea'].fillna(dataset.BuildingArea.mean())
**Drop NA values of Price, since it's our predictive variable we won't impute it**
dataset.dropna(inplace=True)
dataset.shape
#### Let's one hot encode the categorical features
dataset = pd.get_dummies(dataset, drop_first=True)
dataset.head()
#### Let's bifurcate our dataset into train and test dataset
X = dataset.drop('Price', axis=1)
y = dataset['Price']
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=2)
#### Let's train our Linear Regression Model on training dataset and check the accuracy on test set
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(train_X, train_y)
reg.score(test_X, test_y)
reg.score(train_X, train_y)
**Here training score is 68% but test score is 13.85% which is very low**
<h4 style='color:purple'>Normal Regression is clearly overfitting the data, let's try other models</h4>
#### Using Lasso (L1 Regularized) Regression Model
from sklearn import linear_model
lasso_reg = linear_model.Lasso(alpha=50, max_iter=100, tol=0.1)
lasso_reg.fit(train_X, train_y)
lasso_reg.score(test_X, test_y)
lasso_reg.score(train_X, train_y)
#### Using Ridge (L2 Regularized) Regression Model
from sklearn.linear_model import Ridge
ridge_reg= Ridge(alpha=50, max_iter=100, tol=0.1)
ridge_reg.fit(train_X, train_y)
ridge_reg.score(test_X, test_y)
ridge_reg.score(train_X, train_y)
**We see that Lasso and Ridge Regularizations prove to be beneficial when our Simple Linear Regression Model overfits. These results may not be that contrast but significant in most cases.Also that L1 & L2 Regularizations are used in Neural Networks too**

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error

# Load a sample dataset
data = load_boston()
X, y = data.data, data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Ridge regression model with a specific alpha (lambda) value
ridge_model = Ridge(alpha=1.0)

# Fit the model to the training data
ridge_model.fit(X_train, y_train)

# Predict on the test data
y_pred = ridge_model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")