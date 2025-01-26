# Need to add an outlier to see what the metrics are like with them.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score, 
    median_absolute_error, 
    explained_variance_score
)

# Sample true values and predictions for two models
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred_model1 = np.array([2.5, 0.0, 2.0, 8.0])
y_pred_model2 = np.array([3.1, -0.4, 1.8, 6.5])

# Function to compute metrics
def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R-squared": r2_score(y_true, y_pred),
        "Median Absolute Error": median_absolute_error(y_true, y_pred),
        "Explained Variance Score": explained_variance_score(y_true, y_pred),
    }

# Compute metrics for both models
metrics_model1 = compute_metrics(y_true, y_pred_model1)
metrics_model2 = compute_metrics(y_true, y_pred_model2)

# Create a DataFrame to display the results
metrics = ["MAE", "MSE", "RMSE", "R-squared", "Median Absolute Error", "Explained Variance Score"]
data = {
    "Metric": metrics,
    "Model 1": [metrics_model1[m] for m in metrics],
    "Model 2": [metrics_model2[m] for m in metrics],
}

df_metrics = pd.DataFrame(data)

# Display the metrics table
print("Metrics Comparison:\n", df_metrics)

# Plot the true vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_true)), y_true, label="True Values", color="black", marker="o")
plt.plot(range(len(y_true)), y_pred_model1, label="Predictions (Model 1)", linestyle="--", marker="s", color="blue")
plt.plot(range(len(y_true)), y_pred_model2, label="Predictions (Model 2)", linestyle="--", marker="d", color="green")
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("True vs Predicted Values for Two Models")
plt.legend()
plt.grid()
plt.show()
