"""
Forward-Chaining Cross-Validation for Exponential Smoothing Forecasting

This script demonstrates how to:
1. Generate a synthetic time series dataset that mimics hourly usage data with daily
   and weekly seasonal patterns plus random noise.
2. Fit an Exponential Smoothing model to the time series.
3. Evaluate the model using a **forward-chaining (rolling-origin) cross-validation** approach.
   - The training window grows over time, respecting temporal ordering.
   - Each test point uses only historical data to avoid leakage from the future.
4. Compute evaluation metrics for each test point: MAE, RMSE, and MAPE.
5. Visualize the full time series and the forecast for the next point after training on
   all available past data.

Purpose:
- To show how time series forecasting models can be evaluated rigorously.
- To illustrate forward-chaining CV, which is suitable for temporal data where
  conventional random train-test splits would introduce data leakage.
- To provide a template for evaluating and understanding forecast accuracy in
  datasets with seasonal patterns, similar to telecom usage or IoT device data.


In your forward_chaining_cv function, the metrics (MAE, RMSE, MAPE) are computed for each
 forecast window and stored in lists, then averaged at the end.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# ----------------- SYNTHETIC DATA -----------------
np.random.seed(42)
date_range = pd.date_range(start="2025-01-01", periods=24*30, freq="H")  # 30 days hourly data

# simulate daily + weekly pattern with noise
daily_pattern = np.sin(2 * np.pi * date_range.hour / 24)
weekly_pattern = np.sin(2 * np.pi * date_range.dayofweek / 7)
noise = np.random.normal(0, 0.2, len(date_range))

usage = 5 + 2 * daily_pattern + 1.5 * weekly_pattern + noise

df = pd.DataFrame({"Time": date_range, "Usage": usage})
df.set_index("Time", inplace=True)

# ----------------- FORWARD-CHAINING CROSS-VALIDATION -----------------
def forward_chaining_cv(y, initial_train_size=168, seasonal_periods=24):
    """
    Forward-chaining / rolling-origin cross-validation
    y: time series pd.Series
    initial_train_size: number of periods for first training window
    seasonal_periods: seasonality for Exponential Smoothing
    """
    n = len(y)
    mae_list, rmse_list, mape_list = [], [], []

    # Start rolling forecast after initial training window
    for i in range(initial_train_size, n):
        train = y.iloc[:i]
        test = y.iloc[i:i+1]

        model = ExponentialSmoothing(
            train,
            trend='add',
            seasonal='add',
            seasonal_periods=seasonal_periods
        ).fit()

        forecast = model.forecast(1)

        mae_list.append(mean_absolute_error(test, forecast))
        rmse_list.append(np.sqrt(mean_squared_error(test, forecast)))
        mape_list.append(np.mean(np.abs((test - forecast) / test)) * 100)

    print(f"Forward-Chaining CV Metrics:")
    print(f"MAE: {np.mean(mae_list):.3f}, RMSE: {np.mean(rmse_list):.3f}, MAPE: {np.mean(mape_list):.2f}%")

    return mae_list, rmse_list, mape_list

# ----------------- RUN CROSS-VALIDATION -----------------
mae_list, rmse_list, mape_list = forward_chaining_cv(df['Usage'], initial_train_size=168, seasonal_periods=24)

# ----------------- VISUALIZE LAST FORECAST -----------------
train_full = df['Usage']
model_full = ExponentialSmoothing(train_full, trend='add', seasonal='add', seasonal_periods=24).fit()
forecast_5days = model_full.forecast(24*5)  # 5 days

plt.figure(figsize=(12,4))
plt.plot(df.index, df['Usage'], label='Full Series')
plt.plot(pd.date_range(df.index[-1]+pd.Timedelta(hours=1), periods=24*5, freq='H'),
         forecast_5days, color='red', linestyle='--', label='5-Day Forecast')
plt.legend()
plt.title("Exponential Smoothing - 5-Day Forecast")
plt.show()

print(rmse_list)
print(mape_list)