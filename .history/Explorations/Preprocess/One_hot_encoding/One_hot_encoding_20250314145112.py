import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the dataset
df = pd.read_csv("..\..\..\Datasets\homeprices.csv")
print("Dataset Loaded:\n", df.head())

print("\nI think we are testing the effects of different types of one-hot encoding.\n")

# Create dummy variables for the 'town' column
dummies = pd.get_dummies(df.town)
print("\nDummy Variables Created for 'town' column:")
print(dummies.columns)

# Merge dummy variables with the original dataframe
merged = pd.concat([df, dummies], axis='columns')
print("\nMerged DataFrame with Dummy Variables:")
print(merged.head())

# Drop the 'town' column
final = merged.drop(['town'], axis='columns')
print("\nDataFrame After Dropping 'town' Column:")
print(final.head())

# Drop one of the dummy variables to avoid the dummy variable trap
final = final.drop(['west windsor'], axis='columns')
print("\nDataFrame After Dropping 'west windsor' Dummy Variable:")
print(final.head())

# Split the dataset into features (X) and target (y)
X = final.drop('price', axis='columns')
y = final.price

# Initialize and train a Linear Regression model
model = LinearRegression()
model.fit(X, y)
print("\nLinear Regression Model Trained.")

# Predict house prices
print("\nPredicted House Prices for all data points:")
predicted_prices = model.predict(X)
print(predicted_prices)

# Get the model's accuracy
print("\nModel Accuracy (R^2 score):")
print(model.score(X, y))

print("\nPredicting for a specific house (3400 sq ft in West Windsor):")
example1=model.predict([[3400, 0, 0]])
print(example1)  # 3400 sqr ft home in West Windsor
example2=model.predict([[2800, 0, 1]])
print("\nPredicting for another specific house (2800 sq ft in Robbinsville):")
print(example2)  # 2800 sqr ft home in Robbinsville

# Use LabelEncoder to convert town names into numbers
print("\nUsing LabelEncoder to convert town names into numbers:")
le = LabelEncoder()
dfle = df.copy()
dfle.town = le.fit_transform(dfle.town)
# Print which name is encoded to which number
print("Town names and their corresponding numbers:", dict(zip(le.classes_, le.transform(le.classes_))))
print("\nTransformed DataFrame with Encoded Town Names:")
print(dfle.head())

print("\nDefine the feature set X and target variable y after Label Encoding:")
X = dfle[['town', 'area']].values
y = dfle.price.values

print("\nApplying OneHotEncoder for the 'town' column:")
ct = ColumnTransformer([('town', OneHotEncoder(), [0])], remainder='passthrough')
X = ct.fit_transform(X)
print("Transformed Features after OneHotEncoding:\n", X)

print("\nDropping the first column to avoid the dummy variable trap:")
X = X[:, 1:]
print("Features After Dropping the First Column:\n", X)

# Train the Linear Regression model with the transformed features
model.fit(X, y)
print("\nLinear Regression Model Trained on OneHotEncoded Features.")

# Make predictions with the transformed data
print("\nPredicting for a specific house (3400 sq ft in West Windsor) with OneHotEncoded data:")
example1_onehot=model.predict([[0, 1, 0, 3400]])
print(example1_onehot)  # 3400 sqr ft home in West Windsor
print("\nPredicting for another specific house (2800 sq ft in Robbinsville) with OneHotEncoded data:")
example2_onehot=model.predict([[0, 0, 1, 2800]])
print(example2_onehot)  # 2800 sqr ft home in Robbinsville

# comparision

