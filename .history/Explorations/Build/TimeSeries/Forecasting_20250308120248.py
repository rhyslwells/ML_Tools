import numpy as np
import pandas as pd
import plotly.graph_objs as go
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

# Download MSFT Stock Data
ticker = "MSFT"
msft = yf.download(ticker, start="2020-01-01", end="2024-01-01")

# Prepare Data
msft = msft[['Adj Close']].rename(columns={'Adj Close': 'value'})
msft.index.name = 'timestamp'

# Utility Functions
def plt_forecast(predictions, fc_method, train, test):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train['value'], mode='lines', name="Train"))
    fig.add_trace(go.Scatter(x=test.index, y=test['value'], mode='lines', name="Validation"))
    fig.add_trace(go.Scatter(x=predictions.index, y=predictions, mode='lines', name="Baseline Forecast"))
    
    fig.update_layout(yaxis_title='Value', xaxis_title='Date',
                      title=f'Baseline Forecasting using {fc_method}')
    fig.show()

def fcast_evaluation(predicted, actual):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# Cross Validation Setup
tscv = TimeSeriesSplit(n_splits=5)

# Mean Forecasting
mean_results_df = pd.DataFrame(columns=['MSE', 'MAE', 'RMSE', 'MAPE'])

for train_index, test_index in tscv.split(msft):
    train_df, test_df = msft.iloc[train_index], msft.iloc[test_index]
    mean_predictions = pd.Series(np.full(len(test_df), train_df['value'].mean()), index=test_df.index)
    
    mean_results_df = mean_results_df.append(fcast_evaluation(mean_predictions.values, test_df['value'].values), ignore_index=True)

print("Mean Forecasting Results:\n", mean_results_df)

# Naive Forecasting
naive_results_df = pd.DataFrame(columns=['MSE', 'MAE', 'RMSE', 'MAPE'])

for train_index, test_index in tscv.split(msft):
    train_df, test_df = msft.iloc[train_index], msft.iloc[test_index]
    naive_predictions = pd.Series(np.full(len(test_df), train_df['value'].iloc[-1]), index=test_df.index)
    
    naive_results_df = naive_results_df.append(fcast_evaluation(naive_predictions.values, test_df['value'].values), ignore_index=True)

print("Naive Forecasting Results:\n", naive_results_df)

# Seasonal Naive Forecasting
snaive_results_df = pd.DataFrame(columns=['MSE', 'MAE', 'RMSE', 'MAPE'])

for train_index, test_index in tscv.split(msft):
    train_df, test_df = msft.iloc[train_index], msft.iloc[test_index]
    last_7_days_values = train_df['value'].iloc[-7:].values
    snaive_predictions = pd.Series([last_7_days_values[i % 7] for i in range(len(test_df))], index=test_df.index)
    
    snaive_results_df = snaive_results_df.append(fcast_evaluation(snaive_predictions.values, test_df['value'].values), ignore_index=True)

print("Seasonal Naive Forecasting Results:\n", snaive_results_df)
