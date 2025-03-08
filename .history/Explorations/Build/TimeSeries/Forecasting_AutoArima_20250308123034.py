# forecasting_AutoArima.py

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import yfinance as yf

# Helper functions

import pmdarima as pm
from pmdarima.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Load/split your data
y = pm.datasets.load_wineind()
train, test = train_test_split(y, train_size=150)

# Fit your model
model = pm.auto_arima(train, seasonal=True, m=12)

# make your forecasts
forecasts = model.predict(test.shape[0])  # predict N steps into the future

# Visualize the forecasts (blue=train, green=forecasts)
x = np.arange(y.shape[0])
plt.plot(x[:150], train, c='blue')
plt.plot(x[150:], forecasts, c='green')
plt.show()



def adf_test(series):
    """
    Perform Augmented Dickey-Fuller test for stationarity.
    """
    result = adfuller(series.dropna())
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Critical Values:')
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
raw_data = pd.read_csv('../../data/msft_stat.csv', index_col=0)
raw_data.index = pd.to_datetime(raw_data.index)
raw_data = raw_data.asfreq('D')

# Box-Cox Transformation
lam = 0.40136288976843615
raw_data['Close_boxcox'] = boxcox(raw_data['Close'], lam)

# Perform ADF test for stationarity
adf_test(raw_data['Close_boxcox'])

# Train/Test Split
tscv = TimeSeriesSplit(n_splits=5)

# Auto-ARIMA Forecasting
arima_results_df = pd.DataFrame(columns=['MSE', 'MAE', 'RMSE', 'MAPE'], index=range(1, 6))

for i, (train_idx, test_idx) in enumerate(tscv.split(raw_data)):
    train_df, test_df = raw_data.iloc[train_idx], raw_data.iloc[test_idx]
    y = train_df['Close_boxcox']
    
    arima_model = pm.auto_arima(y, seasonal=False, stepwise=True)
    pred_y = inv_boxcox(arima_model.predict(n_periods=len(test_df)), lam)
    
    print(f"Fold {i+1}:")
    plt_forecast(pred_y, fc_method='ARIMA', train=train_df, test=test_df)
    print(arima_model.summary())
    
    arima_results_df.iloc[i] = fcast_evaluation(pred_y.values, test_df['Close'].values)

# Auto-SARIMA Forecasting
sarima_results_df = pd.DataFrame(columns=['MSE', 'MAE', 'RMSE', 'MAPE'], index=range(1, 6))

for i, (train_idx, test_idx) in enumerate(tscv.split(raw_data)):
    train_df, test_df = raw_data.iloc[train_idx], raw_data.iloc[test_idx]
    y = train_df['Close_boxcox']
    
    sarima_model = pm.auto_arima(y, seasonal=True, stepwise=True)
    pred_y = inv_boxcox(sarima_model.predict(n_periods=len(test_df)), lam)
    
    print(f"Fold {i+1}:")
    plt_forecast(pred_y, fc_method='SARIMA', train=train_df, test=test_df)
    print(sarima_model.summary())
    
    sarima_results_df.iloc[i] = fcast_evaluation(pred_y.values, test_df['Close'].values)
