
# ----
# # Exponential Smoothing
# -----
# 
# Exponential Smoothing is a forecasting method that applies greater weight to more recent observations while exponentially decreasing the weight of older data points. In this notebook, I will explore and compare the three exponential smoothing methods using the **holtwinters** package in Python:
# 
# **1. Simple Exponential Smoothing:**
# 
# This method is used for time series data without trend or seasonality. 
# 
# It averages past data, giving more weight to recent observations and less to older ones. The smoothing parameter adjusts how much emphasis is placed on recent data and is used to set the level. 
# 
# **2. Holt’s Linear Trend Model (Double Exponential Smoothing):**
# 
# This method builds on simple exponential smoothing by adding a trend component (another paramter). This method provides predictions that take into account the level and direction of trend.
# 
# **3. Holt-Winters Seasonal Model (Triple Exponential Smoothing):**
# 
# This model builds on the Holt's Linear Model by adding a seasonality component. This method provides predictions that take into account the level and direction of trend as well as the observed seasonality in data.


# ## Set Up
# ---


import numpy as np
import pandas as pd


import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.api import tsa 
import statsmodels.api as sm

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


# ## Utility Functions
# ----


def plt_forecast(predictions, fc_method, train, test):
    """
    Description:
    Plots the training data, validation data (actual) and baseline predictions on a single graph.

    Parameters:
    - predictions : A Series containing the predicted values with date indices.
    - fc_method: A string describing the forecasting method used.
    - train: A dataframe with train data.
    - test: A dataframe with test data

    Output:
    The function creates a plot using Plotly to visualise:
        - Training data
        - Validation data 
        - Baseline forecast predictions

    """
    
    # Plot to visualise the training data, test data and baseline prediction
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name="Train"))
    fig.add_trace(go.Scatter(x=test.index, y=test['Close'], mode='lines', name="Validation"))
    fig.add_trace(go.Scatter(x=predictions.index, y=predictions, mode='lines', name="Baseline Forecast"))

    fig.update_layout(
        yaxis_title='Close', 
        xaxis_title='Date',
        title= f'{fc_method}'
    )
    fig.show()


def fcast_evaluation(predicted, actual):
    """
    Description:
    To evaluate forecasting performance using multiple metrics.

    Parameters:
    predicted: Forecasted values.
    actual: Actual observed values.

    Output:
    A dictionary containing the evaluation metrics:
        - 'MSE': Mean Squared Error
        - 'MAE': Mean Absolute Error
        - 'RMSE': Root Mean Squared Error
        - 'MAPE': Mean Absolute Percentage Error
    """

    err= actual - predicted

    # Calculating MSE
    mse = mean_squared_error(actual, predicted)

    # Calculating MAE
    mae = mean_absolute_error(actual, predicted)

    # Calculating RMSE
    rmse = np.sqrt(mse)

    # Calculating MAPE
    abs_percent_err = np.abs(err/actual)
    mape = abs_percent_err.mean() * 100

    return {'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
            }


# ## Data Loading
# ----


raw_data = pd.read_csv('../../data/msft_cleaned.csv', index_col=0)


raw_data.index = pd.to_datetime(raw_data.index)


raw_data = raw_data.asfreq('D')


raw_data.head(5)


# ## Cross Validation Train/Test Split 
# ----


tscv = TimeSeriesSplit(n_splits=5)


# ## Simple Exponential Smoothing
# ---


# Creating a Dataframe to store the results
simple_results_df = pd.DataFrame(columns = ['MSE', 'MAE', 'RMSE', 'MAPE','Smoothing Level'], index = [1,2,3,4,5])


for i, (train_index, test_index) in enumerate(tscv.split(raw_data)):
    
    # Create train/test datraframes using the indexes from tcsv
    train_df = raw_data.iloc[train_index, :]
    test_df = raw_data.iloc[test_index, :]

    simp_exp_smooth = SimpleExpSmoothing(train_df['Close'])
    simp_model = simp_exp_smooth.fit(optimized = True)
    forecasts = simp_model.forecast(test_df.shape[0])

    # Plotting the forecasted values alongside actual data and previous data (training data) 
    print(f"Fold {i+1}:") 
    plt_forecast(forecasts, fc_method='Simple Exponential Smoothing', train=train_df, test= test_df)
    
    # Adding smoothing level param to the results dict
    results_dict = fcast_evaluation(forecasts.values, test_df['Close'].values)
    results_dict['Smoothing Level'] = simp_model.params['smoothing_level']
    simple_results_df.iloc[i] = results_dict


# ### Evaluation


simple_results_df


# -----
# **Comment:**
# 
# The  smoothing_level parameter across all folds is relatively close to 1, ranging from 0.745 to 0.970. This suggests that the this model is heavily relying on the most recent data points for its forecasts, similar to the naive baseline method in the previous notebook. 
# 
# Fold 4 seems to perform the worst, likely due to the high volatility seen in the close price. The model's smoothing parameter is quite high, indicating that it is overly focused on the most recent data points. As a result, it fails to capture the abrupt changes in the data, since it emphasises the immediate data points too much.
# 
# Fold 1 performs the best. When examining the data points, they appear relatively stable and in this case, having a high smoothing parameter is beneficial because the recent data is consistent, allowing the model to make more accurate predictions.
# 
# Overall, Simple Exponential Smoothing did not provide much improvement in forecasting accuracy over the baseline methods. The model appears to handle stable  data much better than data with high variability (as seen in the difference in perofmrance between Fold 1 and Fold 4). This suggests that more complex methods may be needed to deal with periods of high volatility.
# 


# ## Holt's Linear Trend Model
# ----


# Creating a Dataframe to store the results
holt_lin_results_df = pd.DataFrame(columns = ['MSE', 'MAE', 'RMSE', 'MAPE','Smoothing Level', 'Smoothing Trend'], index = [1,2,3,4,5])


for i, (train_index, test_index) in enumerate(tscv.split(raw_data)):
    
    # Create train/test datraframes using the indexes from tcsv
    train_df = raw_data.iloc[train_index, :]
    test_df = raw_data.iloc[test_index, :]

    holt_lin = Holt(train_df['Close'], damped_trend=True)
    holt_lin_model = holt_lin.fit(optimized = True)
    forecast_2 = holt_lin_model.forecast(test_df.shape[0])

    # Plotting the forecasted values alongside actual data and previous data (training data) 
    print(f"Fold {i+1}:") 
    plt_forecast(forecast_2, fc_method='Holt\'s Linear Trend Model', train=train_df, test= test_df)
    
    results_dict = fcast_evaluation(forecast_2.values, test_df['Close'].values)
    results_dict['Smoothing Level'] = holt_lin_model.params['smoothing_level']
    results_dict['Smoothing Trend'] = holt_lin_model.params['smoothing_trend']
    holt_lin_results_df.iloc[i] = results_dict


holt_lin_results_df


# -----
# **Comment:**
# 
# The smoothing_level (emphasis on recent data points) and smoothing_trend (model's ability to capture trend changes over time) parameters in Holt's Linear Trend method show significant variation across folds. The smoothing level ranges from 0.679 to 0.970, but the smoothing_trend values are very low, especially in Folds 4 and 5 where it is 0.0, indicating hardly any emphasis on the trend.
# 
# Fold 4 again has the worst performance with MSE of 6024.53 and MAPE of 21.82%. This is in line with findings from simple exponential method, the high smoothing level and zero trend parameter suggest the model is heavily focused on recent data without accounting for trends and so fails to capture the high volatility in this fold, leading to poor forecasts.
# 
# Overall, the model's forecasts are horizontal lines across folds, suggesting that Holt’s method is not effectively capturing trends in the stock data. This could be due to the trend component being too low, indicating that more complex methods might be needed to better capture trends and improve forecasting accuracy.


# ## Holt Winter's Method
# ----


# Creating a Dataframe to store the results
holt_win_results_df = pd.DataFrame(columns = ['MSE', 'MAE', 'RMSE', 'MAPE','Smoothing Level', 'Smoothing Trend', 'Smoothing Seasonal'], index = [1,2,3,4,5])


for i, (train_index, test_index) in enumerate(tscv.split(raw_data)):
    
    # Create train/test datraframes using the indexes from tcsv
    train_df = raw_data.iloc[train_index, :]
    test_df = raw_data.iloc[test_index, :]

    exp_smooth = ExponentialSmoothing(train_df['Close'],
                                    trend = 'add',
                                    seasonal= 'add',
                                    seasonal_periods= 7)
    holt_win_model = exp_smooth.fit(optimized=True)
    forecast_3 = holt_win_model.forecast(test_df.shape[0])

    # Plotting the forecasted values alongside actual data and previous data (training data) 
    print(f"Fold {i+1}:") 
    plt_forecast(forecast_3, fc_method='Holt Winter\'s Model', train=train_df, test= test_df)
    
    results_dict = fcast_evaluation(forecast_3.values, test_df['Close'].values)
    results_dict['Smoothing Level'] = holt_win_model.params['smoothing_level']
    results_dict['Smoothing Trend'] = holt_win_model.params['smoothing_trend']
    results_dict['Smoothing Seasonal'] = holt_win_model.params['smoothing_seasonal']
    holt_win_results_df.iloc[i] = results_dict


holt_win_results_df


# ----
# **Comment:**
# 
# The smoothing_level, smoothing_trend and smoothing_seasonal parameters in Holt-Winters’ method show varying values across folds. The smoothing_level ranges from 0.731 to 0.969 and indicates high weight is given to recent data points. The smoothing_trend values are very low or zero, ranging from 0.0 to 0.000005, reflecting minimal emphasis on changes in the trend. The smoothing_seasonal parameter is zero across all folds, suggesting no seasonal component was applied. This could be due to the seasonlity in the data being weak.
# 
# Fold 4 performs the worst with MSE of 4011.14 and MAPE of 17.80%. The high smoothing level with zero trend and seasonal parameters indicates the model is overly focused on recent data without accounting for trends or seasonality, and so fails to handle the volatility present in this fold.
# 
# Fold 5 shows relatively good performance with MSE of 728.41 and MAPE of 5.10%. The high smoothing level (0.9687) and very low smoothing trend (0.000005) suggest that the model captures recent changes well but still does not address seasonality.
# 
# Overall, the Holt-Winters method shows some improvement by capturing trends in forecasts. However, the zero values for the seasonal parameter suggest that the data's seasonality may not be strong enough to be effectively captured by this method. 
# 


# ## Conclusion
# -----
# 
# The exponential smoothing methods explored in this notebook did not perform well with the stock data.
# 
# Simple Exponential Smoothing, Holt's Linear Trend and Holt-Winters’ methods showed varying results but overall they all struggled in capturing trends and seasonality effectively.
# 
# Advanced forecasting models such as ARIMA may be needed in order to better capture the complexities seen in this data.


