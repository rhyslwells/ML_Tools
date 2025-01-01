'''This script demonstrates how to apply a GradientBoostingRegressor to predict
 house prices based on certain features, evaluate the model’s performance, 
 and make predictions for new data.
'''

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Create a simple synthetic dataset with house details
# Here 'rooms' is the number of rooms, 'size' is the square footage, and 'price' is the price of the house
data = {
    'rooms': [2, 3, 4, 5, 6],  # Number of rooms in the house
    'size': [800, 1200, 1500, 2000, 2500],  # Size of the house in square feet
    'price': [200000, 300000, 350000, 450000, 550000]  # Price of the house
}

# Convert the dictionary to a pandas DataFrame for easier manipulation
df = pd.DataFrame(data)

# Define the features (X) and target variable (y)
# X includes 'rooms' and 'size' as features for prediction
# y is the target variable 'price' we are trying to predict
X = df[['rooms', 'size']]  # Features: rooms and size
y = df['price']            # Target: price

# Split the dataset into training and testing sets
# 80% of the data is used for training, and 20% is used for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the GradientBoostingRegressor model
# This model will use 100 trees (n_estimators), a learning rate of 0.1, and a maximum tree depth of 3
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)

# Train the model on the training data
# The model learns from the relationship between the features (rooms, size) and the target variable (price)
model.fit(X_train, y_train)

# Use the trained model to make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance by calculating the Mean Squared Error (MSE)
# This will give an indication of how well the model's predictions match the actual prices
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")  # Output the MSE to evaluate performance

# Example: Predict the price for a new house with 3 rooms and 1500 square feet
# This simulates a real-world scenario where you use the trained model to predict unseen data
new_data = np.array([[3, 1500]])  # New house with 3 rooms and 1500 sq ft
predicted_price = model.predict(new_data)

# Output the predicted price for this new house
# The prediction is based on the learned patterns from the training data
print(f"Predicted Price for a house with 3 rooms and 1500 sq ft: ${predicted_price[0]:,.2f}")
