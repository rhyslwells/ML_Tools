# forecasting_Statsmodels.py

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import yfinance as yf

# Helper functions


def adf_test(series):
    """
    Perform Augmented Dickey-Fuller test for stationarity.
    """
    result = adfuller(series.dropna())
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Critical Values: ')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

def plt_forecast(pred_y, fc_method, train, test):
    """
    Plot the forecast results.
    """
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train'))
    fig.add_trace(go.Scatter(x=test.index, y=test['Close'], mode='lines', name='Test'))
    fig.add_trace(go.Scatter(x=test.index, y=pred_y, mode='lines', name=f'{fc_method} Forecast'))
    
    fig.update_layout(title=f"{fc_method} Forecast", xaxis_title="Date", yaxis_title="Price")
    fig.show()

def fcast_evaluation(pred, true):
    """
    Evaluate forecasting model performance.
    """
    mse = mean_squared_error(true, pred)
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((true - pred) / true)) * 100
    return mse, mae, rmse, mape

# Data Loading
ticker = "MSFT"
data = yf.download(ticker, start="2020-01-01", end="2024-01-01")
data = data[['Close']].rename(columns={'Close': 'value'})
data.index.name = 'timestamp'

# Data Preparation and Stationarity Check
raw_data = data.copy()  # Use the downloaded data directly
raw_data = raw_data.asfreq('D')  # Ensure daily frequency

# Optional: Perform any necessary preprocessing, e.g., Box-Cox Transformation
lam = 0.40136288976843615
raw_data['Close_boxcox'] = boxcox(raw_data['value'], lam)

# Perform ADF test for stationarity
adf_test(raw_data['Close_boxcox'])

# Train/Test Split
tscv = TimeSeriesSplit(n_splits=5)

# ARIMA Forecasting using Statsmodels
arima_results_df = pd.DataFrame(columns=['MSE', 'MAE', 'RMSE', 'MAPE'], index=range(1, 6))

for i, (train_idx, test_idx) in enumerate(tscv.split(raw_data)):
    train_df, test_df = raw_data.iloc[train_idx], raw_data.iloc[test_idx]
    y = train_df['Close_boxcox']
    
    # Fit ARIMA model
    arima_model = sm.tsa.ARIMA(y, order=(1, 1, 1))  # Adjust AR, I, MA parameters as needed
    arima_fitted = arima_model.fit()
    
    # Make predictions
    pred_y = inv_boxcox(arima_fitted.forecast(len(test_df)), lam)
    
    print(f"Fold {i+1}:")
    plt_forecast(pred_y, fc_method='ARIMA', train=train_df, test=test_df)
    print(arima_fitted.summary())
    
    arima_results_df.iloc[i] = fcast_evaluation(pred_y.values, test_df['Close'].values)

# SARIMA Forecasting using Statsmodels
sarima_results_df = pd.DataFrame(columns=['MSE', 'MAE', 'RMSE', 'MAPE'], index=range(1, 6))

for i, (train_idx, test_idx) in enumerate(tscv.split(raw_data)):
    train_df, test_df = raw_data.iloc[train_idx], raw_data.iloc[test_idx]
    y = train_df['Close_boxcox']
    
    # Fit SARIMA model
    sarima_model = sm.tsa.SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))  # Adjust parameters as needed
    sarima_fitted = sarima_model.fit()
    
    # Make predictions
    pred_y = inv_boxcox(sarima_fitted.forecast(len(test_df)), lam)
    
    print(f"Fold {i+1}:")
    plt_forecast(pred_y, fc_method='SARIMA', train=train_df, test=test_df)
    print(sarima_fitted.summary())
    
    sarima_results_df.iloc[i] = fcast_evaluation(pred_y.values, test_df['Close'].values)
