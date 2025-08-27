"""
Random Forest Forecasting for Future Time Series (5 Days Ahead, No Leakage)

This script demonstrates:
1. Generating synthetic hourly usage data.
2. Creating lag, rolling, and time-based features.
3. Iteratively predicting 5 days ahead into **future time points**.
4. Avoiding data leakage by never using actual future values.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# ----------------- SYNTHETIC DATA -----------------
np.random.seed(42)
date_range = pd.date_range(start="2025-01-01", periods=24*30, freq="H")  # 30 days
daily_pattern = np.sin(2 * np.pi * date_range.hour / 24)
weekly_pattern = np.sin(2 * np.pi * date_range.dayofweek / 7)
noise = np.random.normal(0, 0.2, len(date_range))
usage = 5 + 2*daily_pattern + 1.5*weekly_pattern + noise

df = pd.DataFrame({"Time": date_range, "Usage": usage}).set_index("Time")
print("Synthetic data ready. Last 5 rows:")
print(df.tail())

# ----------------- FEATURE ENGINEERING FOR TRAINING -----------------
def build_feature_matrix(series, lags=[1,2,3,24], rolling_windows=[3,6,12]):
    X, y = [], []
    for i in range(max(max(lags), max(rolling_windows)), len(series)):
        row = []
        # lag features
        for lag in lags:
            row.append(float(series.iloc[i-lag]))
        # rolling mean features
        for window in rolling_windows:
            row.append(float(series.iloc[i-window:i].mean()))
        # time features
        timestamp = series.index[i]
        row.extend([timestamp.hour, timestamp.dayofweek])
        X.append(row)
        y.append(float(series.iloc[i]))
    return pd.DataFrame(X), pd.Series(y)


X_train, y_train = build_feature_matrix(df)
print(f"\nFeature matrix for training created. Shape: {X_train.shape}")

# ----------------- TRAIN RANDOM FOREST -----------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("Random Forest trained.")

# ----------------- ITERATIVE FORECAST INTO FUTURE -----------------
forecast_horizon = 24*5  # 5 days
lags = [1,2,3,24]
rolling_windows = [3,6,12]

pred_series = df['Usage'].copy()  # start with historical data
forecast_values = []

print("\nStarting 5-day future forecast...")
for step in range(forecast_horizon):
    current_time = pred_series.index[-1] + pd.Timedelta(hours=1)
    
    # Build features for next step
    X_next = []
    for lag in lags:
        X_next.append(pred_series.iloc[-lag])
    for window in rolling_windows:
        X_next.append(pred_series.iloc[-window:].mean())
    # time features
    X_next.extend([current_time.hour, current_time.dayofweek])
    
    # Predict next value
    y_pred = rf.predict([X_next])[0]
    forecast_values.append(y_pred)
    
    # Append predicted value to series for next iteration
    pred_series = pd.concat([pred_series, pd.Series([y_pred], index=[current_time])])
    
    if step < 5:  # print first few steps
        print(f"Step {step+1}, Forecasted value: {y_pred:.3f}, Time: {current_time}")

# ----------------- VISUALIZATION -----------------
plt.figure(figsize=(12,4))
plt.plot(df.index, df['Usage'], label='Historical Usage')
forecast_index = pd.date_range(df.index[-1]+pd.Timedelta(hours=1), periods=forecast_horizon, freq='H')
plt.plot(forecast_index, forecast_values, color='red', linestyle='--', label='5-Day Forecast')
plt.legend()
plt.title("Random Forest - Future 5-Day Forecast (No Leakage)")
plt.show()
print("Forecast complete.")
