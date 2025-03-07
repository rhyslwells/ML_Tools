# Baseline Forecasting Script
# 
# This script performs baseline time series forecasting using three methods:
# 1. **Mean Forecasting**: Predicts future values using the mean of the training set.
# 2. **Naive Forecasting**: Uses the last observed value in the training set as the forecast.
# 3. **Seasonal Naive Forecasting**: Uses values from the last 7 days as the forecast for corresponding future days.
#
# The script uses **TimeSeriesSplit** for cross-validation and evaluates each method using MSE, MAE, RMSE, and MAPE.
# It also visualizes the predictions against actual data using Plotly.

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

# Load Data
raw_data = pd.read_csv('../../../Datasets/NAB-TimeSeries-Data/artificialNoAnomaly/art_daily_no_noise.csv')
#no anomalys
Datasets\NAB-TimeSeries-Data\artificialNoAnomaly\art_daily_perfect_square_wave.csv
Datasets\NAB-TimeSeries-Data\artificialNoAnomaly\art_daily_small_noise.csv
Datasets\NAB-TimeSeries-Data\artificialNoAnomaly\art_flatline.csv
Datasets\NAB-TimeSeries-Data\artificialNoAnomaly\art_noisy.csv
# with anomalys
Datasets\NAB-TimeSeries-Data\artificialWithAnomaly\art_daily_flatmiddle.csv
Datasets\NAB-TimeSeries-Data\artificialWithAnomaly\art_daily_jumpsdown.csv
Datasets\NAB-TimeSeries-Data\artificialWithAnomaly\art_daily_jumpsup.csv
Datasets\NAB-TimeSeries-Data\artificialWithAnomaly\art_daily_nojump.csv
Datasets\NAB-TimeSeries-Data\artificialWithAnomaly\art_increase_spike_density.csv
Datasets\NAB-TimeSeries-Data\artificialWithAnomaly\art_load_balancer_spikes.csv



raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'])
raw_data.set_index('timestamp', inplace=True)

# Utility Functions
def plt_forecast(predictions, fc_method, train, test):
    """
    Plots the training data, validation data (actual), and baseline predictions on a single graph.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train['value'], mode='lines', name="Train"))
    fig.add_trace(go.Scatter(x=test.index, y=test['value'], mode='lines', name="Validation"))
    fig.add_trace(go.Scatter(x=predictions.index, y=predictions, mode='lines', name="Baseline Forecast"))
    
    fig.update_layout(yaxis_title='Value', xaxis_title='Date',
                      title=f'Baseline Forecasting using {fc_method}')
    fig.show()

def fcast_evaluation(predicted, actual):
    """
    Evaluates forecasting performance using multiple metrics.
    """
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = (np.abs(actual - predicted) / actual).mean() * 100
    
    return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# Cross Validation Setup
tscv = TimeSeriesSplit(n_splits=5)

# Mean Forecasting
mean_results_df = pd.DataFrame(columns=['MSE', 'MAE', 'RMSE', 'MAPE'], index=range(1, 6))

for i, (train_index, test_index) in enumerate(tscv.split(raw_data)):
    train_df, test_df = raw_data.iloc[train_index, :], raw_data.iloc[test_index, :]
    baseline_pred = np.full(test_df.shape[0], np.mean(train_df['value']))
    mean_predictions = pd.Series(data=baseline_pred, index=test_df.index)
    
    # print(f"Fold {i+1}:")
    # plt_forecast(mean_predictions, fc_method='Mean', train=train_df, test=test_df)
    
    mean_results_df.iloc[i] = fcast_evaluation(mean_predictions.values, test_df['value'].values)

# Print Mean Forecasting Results
print(mean_results_df)

# Naive Forecasting
naive_results_df = pd.DataFrame(columns=['MSE', 'MAE', 'RMSE', 'MAPE'], index=range(1, 6))

for i, (train_index, test_index) in enumerate(tscv.split(raw_data)):
    train_df, test_df = raw_data.iloc[train_index, :], raw_data.iloc[test_index, :]
    baseline_pred = np.full(test_df.shape[0], train_df['value'].iloc[-1])
    naive_predictions = pd.Series(data=baseline_pred, index=test_df.index)
    
    print(f"Fold {i+1}:")
    plt_forecast(naive_predictions, fc_method='Naive', train=train_df, test=test_df)
    
    naive_results_df.iloc[i] = fcast_evaluation(naive_predictions.values, test_df['value'].values)

# Print Naive Forecasting Results
print(naive_results_df)

# Seasonal Naive Forecasting
snaive_results_df = pd.DataFrame(columns=['MSE', 'MAE', 'RMSE', 'MAPE'], index=range(1, 6))

for i, (train_index, test_index) in enumerate(tscv.split(raw_data)):
    train_df, test_df = raw_data.iloc[train_index, :], raw_data.iloc[test_index, :]
    
    last_7_days_values = train_df['value'].iloc[-7:].values
    snaive_fcasts = [last_7_days_values[idx % 7] for idx in range(len(test_index))]
    snaive_predictions = pd.Series(data=snaive_fcasts, index=test_df.index)
    
    print(f"Fold {i+1}:")
    plt_forecast(snaive_predictions, fc_method='Seasonal Naive', train=train_df, test=test_df)
    
    snaive_results_df.iloc[i] = fcast_evaluation(snaive_predictions.values, test_df['value'].values)

# Print Seasonal Naive Forecasting Resultsk
print(snaive_results_df)
