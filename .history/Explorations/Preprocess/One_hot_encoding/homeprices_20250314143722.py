import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the dataset
df = pd.read_csv("..\..\Datasets\homeprices.csv")
print(df)

# Create dummy variables for the 'town' column
dummies = pd.get_dummies(df.town)
print(dummies)

# Merge dummy variables with the original dataframe
merged = pd.concat([df, dummies], axis='columns')
print(merged)

# Drop the 'town' column
final = merged.drop(['town'], axis='columns')
print(final)

# Drop one of the dummy variables to avoid the dummy variable trap
final = final.drop(['west windsor'], axis='columns')
print(final)

# Split the dataset into features (X) and target (y)
X = final.drop('price', axis='columns')
y = final.price

# Initialize and train a Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Predict house prices
print(model.predict(X))  # Predicting for all the data points

# Get the model's accuracy
print(model.score(X, y))

# Predict for a specific house (3400 sq ft in West Windsor)
print(model.predict([[3400, 0, 0]]))  # 3400 sqr ft home in West Windsor

# Predict for another specific house (2800 sq ft in Robbinsville)
print(model.predict([[2800, 0, 1]]))  # 2800 sqr ft home in Robbinsville

# Use LabelEncoder to convert town names into numbers
le = LabelEncoder()
dfle = df.copy()
dfle.town = le.fit_transform(dfle.town)
print(dfle)

# Define the feature set X and target variable y
X = dfle[['town', 'area']].values
y = dfle.price.values

# Apply OneHotEncoder for the 'town' column
ct = ColumnTransformer([('town', OneHotEncoder(), [0])], remainder='passthrough')
X = ct.fit_transform(X)
print(X)

# Drop the first column to avoid the dummy variable trap
X = X[:, 1:]
print(X)

# Train the Linear Regression model with the transformed features
model.fit(X, y)

# Make predictions with the transformed data
print(model.predict([[0, 1, 3400]]))  # 3400 sqr ft home in West Windsor
print(model.predict([[1, 0, 2800]]))  # 2800 sqr ft home in Robbinsville
