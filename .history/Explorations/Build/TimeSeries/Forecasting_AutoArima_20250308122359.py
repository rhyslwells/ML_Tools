# Auto-ARIMA Forecasting

## Introduction
This notebook explores time series forecasting of stock returns using ARIMA, SARIMA, and SARIMAX models. The `pmdarima` package is utilized to simplify the implementation and tuning of these models, which is particularly useful for forecasting stock-related data due to its high volatility.

## Notebook Overview
The goal is to predict the closing price of Microsoft stock for the next 7 days. The workflow consists of the following steps:

1. **Data Loading and Preparation**: Load stock data.
2. **Forecasting with Auto ARIMA**: Use `pmdarima` to automatically identify the best parameters for ARIMA forecasting.
3. **Incorporate Seasonality with SARIMA**: Extend ARIMA to capture seasonal effects.
4. **Enhance Forecasting with SARIMAX**: Incorporate exogenous variables such as interest rates to improve accuracy.
5. **Evaluate Model Performance**: Assess forecast accuracy and analyze residuals to ensure effectiveness.

---

## Setup

```python
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
```

---

## Data Loading

```python
import yfinance as yf

ticker = "MSFT"
data = yf.download(ticker, start="2020-01-01", end="2024-01-01")
data = data[['Close']].rename(columns={'Close': 'value'})
data.index.name = 'timestamp'
```

---

## Data Preparation and Stationarity Check

```python
raw_data = pd.read_csv('../../data/msft_stat.csv', index_col=0)
raw_data.index = pd.to_datetime(raw_data.index)
raw_data = raw_data.asfreq('D')

adf_test(raw_data['Close_boxcox'])
```

---

## Train/Test Split

```python
tscv = TimeSeriesSplit(n_splits=5)
```

---

## Auto-ARIMA Forecasting

```python
from pmdarima import auto_arima

lam = 0.40136288976843615
arima_results_df = pd.DataFrame(columns=['MSE', 'MAE', 'RMSE', 'MAPE'], index=range(1, 6))

for i, (train_idx, test_idx) in enumerate(tscv.split(raw_data)):
    train_df, test_df = raw_data.iloc[train_idx], raw_data.iloc[test_idx]
    y = train_df['Close_boxcox']
    
    arima_model = auto_arima(y, seasonal=False, stepwise=True)
    pred_y = inv_boxcox(arima_model.predict(n_periods=len(test_df)), lam)
    
    print(f"Fold {i+1}:")
    plt_forecast(pred_y, fc_method='ARIMA', train=train_df, test=test_df)
    print(arima_model.summary())
    
    arima_results_df.iloc[i] = fcast_evaluation(pred_y.values, test_df['Close'].values)
```

---

## Auto-SARIMA Forecasting

```python
sarima_results_df = pd.DataFrame(columns=['MSE', 'MAE', 'RMSE', 'MAPE'], index=range(1, 6))

for i, (train_idx, test_idx) in enumerate(tscv.split(raw_data)):
    train_df, test_df = raw_data.iloc[train_idx], raw_data.iloc[test_idx]
    y = train_df['Close_boxcox']
    
    sarima_model = auto_arima(y, seasonal=True, stepwise=True)
    pred_y = inv_boxcox(sarima_model.predict(n_periods=len(test_df)), lam)
    
    print(f"Fold {i+1}:")
    plt_forecast(pred_y, fc_method='SARIMA', train=train_df, test=test_df)
    print(sarima_model.summary())
    
    sarima_results_df.iloc[i] = fcast_evaluation(pred_y.values, test_df['Close'].values)
```

