import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, explained_variance_score

# Sample true and predicted values
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])

# 1. Mean Absolute Error (MAE)
mae = mean_absolute_error(y_true, y_pred)
print("Mean Absolute Error:", mae)

# 2. Mean Squared Error (MSE)
mse = mean_squared_error(y_true, y_pred)
print("Mean Squared Error:", mse)

# 3. Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

# 4. R-squared (R²)
r2 = r2_score(y_true, y_pred)
print("R-squared:", r2)

# 5. Median Absolute Error
median_abs_err = median_absolute_error(y_true, y_pred)
print("Median Absolute Error:", median_abs_err)

# 6. Explained Variance Score
explained_var = explained_variance_score(y_true, y_pred)
print("Explained Variance Score:", explained_var)

import numpy as np
from sklearn.metrics import r2_score

# Sample true and predicted values
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])

# Number of observations and predictors
n = len(y_true)
p = 1  # Example: 1 predictor

# Calculate R-squared
r2 = r2_score(y_true, y_pred)

# Calculate Adjusted R-squared
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
print("Adjusted R-squared:", adjusted_r2)