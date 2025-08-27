import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------- CONFIG -----------------
CSV_FILE = "data/data-series.csv"  # your time series file
TARGET_COL = "Usage"
WINDOW_SIZE = 24  # hours or periods for rolling evaluation

# ----------------- LOAD DATA -----------------
df = pd.read_csv(CSV_FILE, parse_dates=['Time'], index_col='Time')
y = df[TARGET_COL]

# ----------------- CROSS-VALIDATION -----------------
def rolling_cv_exponential(y, window_size=WINDOW_SIZE, seasonal_periods=None):
    """
    Perform rolling window cross-validation for Exponential Smoothing
    """
    mae_list, rmse_list, mape_list = [], [], []

    for i in range(window_size, len(y)):
        train = y.iloc[i-window_size:i]
        test = y.iloc[i:i+1]  # next-step forecast

        model = ExponentialSmoothing(
            train,
            trend='add',          # additive trend
            seasonal='add' if seasonal_periods else None,
            seasonal_periods=seasonal_periods
        ).fit()

        forecast = model.forecast(1)

        mae_list.append(mean_absolute_error(test, forecast))
        rmse_list.append(np.sqrt(mean_squared_error(test, forecast)))
        mape_list.append(np.mean(np.abs((test - forecast) / test)) * 100)

    print("MAE:", np.mean(mae_list))
    print("RMSE:", np.mean(rmse_list))
    print("MAPE:", np.mean(mape_list))

    return mae_list, rmse_list, mape_list

# ----------------- RUN CV -----------------
# Example: daily seasonality with 24 periods
rolling_cv_exponential(y, window_size=168, seasonal_periods=24)
