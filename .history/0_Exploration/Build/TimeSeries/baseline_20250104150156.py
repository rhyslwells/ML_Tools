
# -----
# # Baseline Forecasting
# -----
# 
# ### Notebook Overview
# 
# In this notebook, I’ll explore four different types of baseline forecasting methods:
# 
# **1. Mean Method**
# 
# Predicts that future values will be the mean of all past observed values.
# 
# **2. Naive Method**
# 
# Predicts that the next value will be the same as the last observed value.
# 
# **3. Seasonal Naive Method**
# 
# Predicts that the next value will be the same as the last observed value from the same seasonal period. This method is effective when there are clear seasonal patterns - like in the data we have there is a clear 7 day trend.
# 
# **4. Drift Method**
# 
# Predicts future values by extending the trend line between the first and last observed data points.
# 
# These methods will serve as a benchmark for time series forecasting models. These baseline models can be used assess the performance of more advanced forecasting model such as ARIMA to see if advanced methods are actually better at forecasting or if they are just as good as the simplest approach. 


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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


# ## Utility Functions
# -----


def plt_forecast(predictions, fc_method, train, test):
    """
    Description:
    Plots the training data, validation data (actual), and baseline predictions on a single graph.

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
        title= f'Baseline Forecasting using {fc_method}'
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
# ---


# **Note:** Raw data is used in baseline forecasting to evaluate how well simple methods perform with real-world data.
# 


raw_data = pd.read_csv('../../data/msft_cleaned.csv', index_col=0)


raw_data.index = pd.to_datetime(raw_data.index)


# ## Cross Validation Train/Test Split 
# ----


tscv = TimeSeriesSplit(n_splits=5)


# ----
# **Comment:**
# 
# Deciding to use Cross Validation Train Test split to ensure no data leakage using the TimeSeriesSplit function.
# 
# When dealing with timeseries data, you cannot use the standard train/test random split as the data must remain in chronological order.


# ## Mean Forecasting 
# ----


# Creating a Dataframe to store the results
mean_results_df = pd.DataFrame(columns = ['MSE', 'MAE', 'RMSE', 'MAPE'], index = [1,2,3,4,5])


for i, (train_index, test_index) in enumerate(tscv.split(raw_data)):
    
    # Create train/test datraframes using the indexes from tcsv
    train_df = raw_data.iloc[train_index, :]
    test_df = raw_data.iloc[test_index, :]

    # Calculating predicted Close price values using the mean of train data
    baseline_pred = np.full(test_df.shape[0], np.mean(train_df['Close']))
    mean_predictions = pd.Series(data=baseline_pred, index=test_df.index)

    # Plotting the forecasted values alongside actual data and previous data (train data) 
    print(f"Fold {i+1}:") 
    plt_forecast(mean_predictions, fc_method='Mean', train=train_df, test= test_df)
    
    # Calculating results for each fold using function and adding results to dataframe
    results_dict = fcast_evaluation(mean_predictions.values, test_df['Close'].values)
    mean_results_df.iloc[i] = results_dict
    


# ----
# **Plot Description:**
# 
# Plots shows baseline forecasting using the mean, future values are predicted based on the mean of the training set.
# 
# In the final fold, it is clear to see that this method fails to capture the upward trend seen in the data since the earlier values are pulling the mean down.


# ### Evaluation


mean_results_df


# -----
# **Comment:**
# 
# The results for the mean baseline forecasting varies across the folds.
# 
# - **Fold 1:** The method performs relatively well with a low MSE (3726.70), MAE (58.84), RMSE (61.05) and MAPE (26.72). This indicates that the mean forecast is reasonably accurate for the first fold.
# 
# - **Fold 2:** The metrics are much worse, with high values of MSE (11248.58), MAE (102.93), RMSE (106.06) and MAPE (34.93). This suggests that the mean method fails to capture the underlying patterns or trends in the data for this fold.
# 
# - **Fold 3:** Improvement in metrics with low MSE (1576.84), MAE (33.76), RMSE (39.71) and MAPE (12.66). This indicates that the mean method aligns better with the data in this fold, providing more accurate forecasts.
# 
# - **Fold 4:** Dip in performance with higher values of MSE (6744.81), MAE (74.67), RMSE (82.13) and MAPE (23.46). While the mean method performs better than in Fold 2, it still does not capture the data's characteristics effectively.
# 
# - **Fold 5:** The method shows poor performance again, with very high values of MSE (27586.40), MAE (164.28), RMSE (166.09) and MAPE (39.93). This suggests that the mean method is unable to adapt to significant changes or trends in the overall data.
# 
# Results of the evaluation metrics show how mean forecasting struggles to deal with the trend and patterns in the data, this could explain the fluctations we see in the metrics across the different folds.


# ## Naive Forecasting
# -----


naive_results_df = pd.DataFrame(columns = ['MSE', 'MAE', 'RMSE', 'MAPE'], index = [1,2,3,4,5])


for i, (train_index, test_index) in enumerate(tscv.split(raw_data)):
    
    train_df = raw_data.iloc[train_index, :]
    test_df = raw_data.iloc[test_index, :]
    
    # Creating baseline predictions, filling array with last observed value
    # Assuming future predictions are equal to the last observed value in the training set  
    baseline_pred = np.full(test_df.shape[0], train_df['Close'].iloc[-1])
    naive_predictions = pd.Series(data=baseline_pred, index=test_df.index)

    print(f"Fold {i+1}:")
    plt_forecast(naive_predictions, fc_method='Naive', train = train_df, test = test_df)
    
    results_dict = fcast_evaluation(naive_predictions.values, test_df['Close'].values)
    naive_results_df.iloc[i] = results_dict


# ----
# **Plot Description:**
# 
# Plots show the baseline forecasting where the prediction for future values is the last value of the train dataset. 
# 
# Here the assumption is that future values are equal to the last historical observation. 
# 
# It is clear from the plots that this method is starting to capture part of the overall trend seen in the data.


# ### Evaluation


naive_results_df


# ----
# **Comment:**
# 
# Overall, the Naive forecasting method also shows inconsistent results across all folds. This again shows limiations of this method in capturing the trend and patterns in the data. 


# ## Seasonal Forecasting
# ----


snaive_results_df = pd.DataFrame(columns = ['MSE', 'MAE', 'RMSE', 'MAPE'], index = [1,2,3,4,5])


for i, (train_index, test_index) in enumerate(tscv.split(raw_data)):

    train_df = raw_data.iloc[train_index, :]
    test_df = raw_data.iloc[test_index, :]

    snaive_fcasts = []

    last_7_days_values = train_df['Close'].iloc[-7:].values
    for idx in range(len(test_index)):
        # forecasts using values from the last 7 days of the training data
        forecast_value = last_7_days_values[idx % 7]
        snaive_fcasts.append(forecast_value)

    print(f"Fold {i + 1}:")
    snaive_predictions = pd.Series(snaive_fcasts, index=test_df.index, name='Predicted')

    plt_forecast(snaive_predictions, 'Seasonal Naive', train=train_df, test=test_df)

    results_dict = fcast_evaluation(snaive_predictions.values, test_df['Close'].values)

    snaive_results_df.iloc[i] = results_dict
    


# ----
# **Plot Description:**
# 
# Plot shows baseline forecasting using a seasonal naive method, future values are predicted based on the repeating seasonlity seen in the past 7 days of the training data.
# 


# ### Evaluation


snaive_results_df


# -----
# **Comment:**
# 
# Overall, the seasonal naive method shows improved performance compared to both the mean and naive methods. 
# 
# By predicting future values based on the same day from the previous week, it effectively captures seasonal patterns. This results in lower error metrics, indicating that it better captures recurring patterns and trends in the data. 


# ## Drift Model
# ----


drift_results_df = pd.DataFrame(columns = ['MSE', 'MAE', 'RMSE', 'MAPE'], index = [1,2,3,4,5])


for i, (train_index, test_index) in enumerate(tscv.split(raw_data)):
   
    train_df = raw_data.iloc[train_index, :]
    test_df = raw_data.iloc[test_index, :]

    # Calculating the drift constant -> slope of a straight line from the first to the last point in the train data
    const = (train_df['Close'].iloc[-1] - train_df['Close'].iloc[0])/(train_df.shape[0] -1)
    fcast_range = range(len(test_index))

    # Making predictions using the last train data point and drift const
    drift_pred = train_df['Close'].iloc[-1] + (fcast_range*const)
    drift_predictions =  pd.Series(drift_pred, index=test_df.index)

    print(f'Fold {i + 1}:')
    plt_forecast(drift_predictions, 'Drift', train=train_df, test=test_df)

    results_dict = fcast_evaluation(drift_predictions.values, test_df['Close'].values)
    drift_results_df.iloc[i] = results_dict
    


# ----
# **Plot Description:**
# 
# Plot shows forecasting using the drift method, future values are predicted based on the trend seen in the training data. The trend in the training data is extended from the last training data point.


# ### Evaluation
# 


drift_results_df


# ----
# **Comment:**
# 
# The Drift forecasting method generally performs well when the data trends are stable and align with the linear trend observed in the training data, as seen in Folds 1 and 5. However, it struggles when there are irregular patterns in the training data which could be why we see higher errors in Folds 2, 3 and 4.
# 
# So far, this method seems to be the best baseline forecasting approach. To be sure, I will calculate the average metrics across all folds for each method and compare them.


# ## Evaluation of Baseline Forecasting Methods
# ----


# To evaluate each of the baseline models, I will be using the following metrics:
# 
# **1. MSE (Mean Squared Error)**
# 
# MSE measures the average of the squares of the errors. Errors are defined as the differences between predicted and actual values
# 
# This metric gives more weight to larger errors and outliers due to the squaring.
# 
# **2. MAE (Mean Absolute Error)**
# 
# MAE measures the average magnitude of errors in predictions regardless of direction (over/under estimation). It is the average of the absolute differences between the predicted and actual values.
# 
# **3. RMSE (Root Mean Squared Error)**
# 
# RMSE measures how much the predictions differ from the actual values.  It calculates the square root of the average squared errors and so is on the same scales as the original data making it more interpretable than MSE.
# 
# **4. MAPE (Mean Absolute Percentage Error)**
# 
# MAPE measures the average percentage error between predicted and actual values.


overall_df = pd.DataFrame(index=['Mean', 'Naive', 'Seasonal Naive', 'Drift'], columns = ['MSE', 'MAE', 'RMSE', 'MAPE'])


mean_metrics = mean_results_df[['MSE', 'MAE', 'RMSE', 'MAPE']].mean()
naive_metrics = naive_results_df[['MSE', 'MAE', 'RMSE', 'MAPE']].mean()
snaive_metrics = snaive_results_df[['MSE', 'MAE', 'RMSE', 'MAPE']].mean()
drift_metrics = drift_results_df[['MSE', 'MAE', 'RMSE', 'MAPE']].mean()


overall_df.loc['Mean'] = mean_metrics
overall_df.loc['Naive'] = naive_metrics
overall_df.loc['Seasonal Naive'] = snaive_metrics
overall_df.loc['Drift'] = drift_metrics


overall_df.sort_values(by = ['MSE', 'MAE', 'RMSE', 'MAPE'], ascending= True)


# #### **Best Baseline: Drift Forecasting Model**
# 
# The Drift method seems to be the best performing baseline forecasting method with the lowest average errors. 
# 
# The Naive method is better than the Seasonal Naive and Mean methods but still less accurate than Drift.


# ## Conclusion
# ----


# These baseline models serve as a benchmark for assessing the performance of more advanced forecasting methods which I am moving on to. 
# 
# By comparing advanced models against these simple methods, we can evaluate whether they offer significant improvements in predictions made or not. 
# 
# After comparing evaluation metrics of all 4 methods, the drift model showed the best performance.Therefore, I will use the drift model as the baseline for comparing more advanced forecasting methods moving forward.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate a mock time series dataset
np.random.seed(42)
time_series = np.random.randn(100).cumsum() + 50  # Random walk

# Convert to a pandas DataFrame
data = pd.DataFrame(time_series, columns=['value'])

# Define forecasting methods
def mean_forecast(series, steps):
    return np.mean(series) * np.ones(steps)

def naive_forecast(series, steps):
    return np.ones(steps) * series.iloc[-1]

def seasonal_naive_forecast(series, steps, season_length):
    return series.iloc[-season_length:].repeat(steps // season_length + 1)[:steps].values

def drift_forecast(series, steps):
    slope = (series.iloc[-1] - series.iloc[0]) / (len(series) - 1)
    return series.iloc[-1] + slope * np.arange(1, steps + 1)

# Forecasting parameters
forecast_steps = 10
season_length = 12  # Assuming monthly data for seasonal naive

# Apply forecasting methods
mean_pred = mean_forecast(data['value'], forecast_steps)
naive_pred = naive_forecast(data['value'], forecast_steps)
seasonal_naive_pred = seasonal_naive_forecast(data['value'], forecast_steps, season_length)
drift_pred = drift_forecast(data['value'], forecast_steps)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(data['value'], label='Actual', color='black')
plt.plot(np.arange(len(data), len(data) + forecast_steps), mean_pred, label='Mean Forecast', linestyle='--')
plt.plot(np.arange(len(data), len(data) + forecast_steps), naive_pred, label='Naive Forecast', linestyle='--')
plt.plot(np.arange(len(data), len(data) + forecast_steps), seasonal_naive_pred, label='Seasonal Naive Forecast', linestyle='--')
plt.plot(np.arange(len(data), len(data) + forecast_steps), drift_pred, label='Drift Forecast', linestyle='--')
plt.legend()
plt.title('Baseline Forecasting Methods')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()