# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

# Utility functions for plotting and evaluation
def plot_forecast(predictions, fc_method, train, test):
    """
    Plots the training data, validation data (actual), and forecast predictions on a single graph.
    """
    plt.figure(figsize=(10,6))
    plt.plot(train.index, train['Close'], label='Training Data')
    plt.plot(test.index, test['Close'], label='Test Data', color='orange')
    plt.plot(predictions.index, predictions, label=f'{fc_method} Predictions', color='green')
    plt.title(f'{fc_method} Forecast')
    plt.legend(loc='best')
    plt.show()

def forecast_evaluation(predicted, actual):
    """
    Evaluates forecast accuracy using MSE, MAE, RMSE, and MAPE.
    """
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# Load the data (Replace with your actual data)
data = pd.read_csv('../../.../Datasets/timeseries_data.csv', parse_dates=True, index_col='Date')
data = data.asfreq('D')  # Set frequency to daily

# Cross-validation split
tscv = TimeSeriesSplit(n_splits=5)

# Creating DataFrame to store results
results_df = pd.DataFrame(columns=['MSE', 'MAE', 'RMSE', 'MAPE'], index=['SES', 'Holt', 'Holt-Winters'])

# Simple Exponential Smoothing
for i, (train_index, test_index) in enumerate(tscv.split(data)):
    train, test = data.iloc[train_index], data.iloc[test_index]
    
    # Apply Simple Exponential Smoothing
    ses_model = SimpleExpSmoothing(train['Close']).fit(optimized=True)
    ses_forecast = ses_model.forecast(test.shape[0])
    
    # Plot forecast
    plot_forecast(ses_forecast, 'Simple Exponential Smoothing', train, test)
    
    # Evaluate forecast
    results = forecast_evaluation(ses_forecast, test['Close'])
    results_df.loc['SES', :] = results

# Holt's Linear Trend (Double Exponential Smoothing)
for i, (train_index, test_index) in enumerate(tscv.split(data)):
    train, test = data.iloc[train_index], data.iloc[test_index]
    
    # Apply Holt’s Linear Trend Model
    holt_model = Holt(train['Close']).fit(optimized=True)
    holt_forecast = holt_model.forecast(test.shape[0])
    
    # Plot forecast
    plot_forecast(holt_forecast, 'Holt\'s Linear Trend', train, test)
    
    # Evaluate forecast
    results = forecast_evaluation(holt_forecast, test['Close'])
    results_df.loc['Holt', :] = results

# Holt-Winters' Method (Triple Exponential Smoothing)
for i, (train_index, test_index) in enumerate(tscv.split(data)):
    train, test = data.iloc[train_index], data.iloc[test_index]
    
    # Apply Holt-Winters Exponential Smoothing
    hw_model = ExponentialSmoothing(train['Close'], trend='add', seasonal='add', seasonal_periods=12).fit(optimized=True)
    hw_forecast = hw_model.forecast(test.shape[0])
    
    # Plot forecast
    plot_forecast(hw_forecast, 'Holt-Winters', train, test)
    
    # Evaluate forecast
    results = forecast_evaluation(hw_forecast, test['Close'])
    results_df.loc['Holt-Winters', :] = results

# Display the results for comparison
print("Comparison of Exponential Smoothing Methods:")
print(results_df)

