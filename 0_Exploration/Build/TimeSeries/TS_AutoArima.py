
# ----
# # Auto-ARIMA Forecasting
# -----
# 
# In this notebook, I explore timeseries forecasting of stock returns using : ARIMA, SARIMA, and SARIMAX. For this analysis, I will be using the pmdarima package. This library simplifies the implementation and tuning of ARIMA and its paramters - this is particularly helpful when forecasting stock related data due to the high volatility. 
# 
# ### Notebook Overview
# 
# The purpose of this project is to predict the close price of Microsfot Stock for the next 7 days. To do this, I will:
# 
# 1. **Data Loading and Preparation:** Load stock data
# 
# 2. **Forecasting with Auto ARIMA:** Use pmdarima to automatically identify the best paramters for ARIMA forecasting.
# 
# 3. **Incorporate Seasonality with SARIMA:** Extend the ARIMA model to capture seasonal effects using pmdarima.
# 
# 4. **Further Enhance Forecasting with SARIMAX:** Use exogenous variables such as interest rates to improve forecasting accuracy.
# 
# 5. **Evaluate Model Performance:** Assess the accuracy of our forecasts and analyse residuals to ensure the model’s effectiveness.
# 


# ## Set Up
# ----


import numpy as np
import pandas as pd

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.api import tsa # time series analysis
import statsmodels.api as sm
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


# ## Utility Functions
# ----


def adf_test(series, threshold = 0.05):
    """
    Description:
    Perform the Augmented Dickey-Fuller (ADF) test to check for stationarity.

    Parameters:
    - series : The time series data to test for stationarity.
    - threshold : The significance level to determine stationarity (default is 0.05).

    Returns:
    - None: Prints the ADF statistic, p-value and stationarity result.
    """
    adf_test = adfuller(series)
    adf_statistic = adf_test[0]
    p_value = adf_test[1]
    
    print(f'ADF Statistic: {adf_statistic}')
    print(f'p-value: {p_value}')
 
    if p_value < threshold:
        print("Data is stationary.")
    else:
        print("Data is not stationary.")


def plt_pred_vs_actual(actual, predicted):
    
    """
    Description:
    Plots the actual versus forecasted Close Price.

    Parameters:
    - actual: DataFrame containing actual Close Price with a DateTime index and a 'Close Price' column.
    - predicted: Series containing forecasted Close Price, with the same index as 'actual'.

    Returns:
    - Displays the plot.
    """

     # Validate inputs
    if not isinstance(actual, pd.DataFrame):
        raise TypeError("Expected 'actual' to be a pandas DataFrame.")
    if not isinstance(predicted, pd.Series):
        raise TypeError("Expected 'predicted' to be a pandas Series with same index (Date) as actual")

    plt.figure(figsize=(10, 5))

    plt.plot(actual.index, actual['Close'], label='Actual Close Price')
    plt.plot(actual.index, predicted, label='Forecasted Close Price', color='red')
    ax = plt.gca()

    ax.spines[['top', 'right']].set_visible(False)
    plt.title('Close Price Forecast vs Actual', fontweight = 'bold', fontsize = '15')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()


def inv_diff(last_observed_value, predicted):
    """
    Description:
    Reverts a differenced time series to its original scale

    Parameters:
    - last_observed_value : The last observed value from the original time series.
    - predicted : Series containing the differenced forecasted values to be restored.

    Returns:
    - restored_series : Series with the restored predicted values 

    """    
    current_value = last_observed_value

    inv_diff_values = []

    for diff in predicted:
        inv_diff_value = current_value + diff
        inv_diff_values.append(inv_diff_value)
        current_value = inv_diff_value  

    restored_series = pd.Series(inv_diff_values, index=predicted.index)

    return restored_series


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
        title= f'{fc_method}'
    )
    fig.show()


# ## Data Loading
# ----


# ARIMA requires input data to be stationary, therefore I will now be working with the dataset which has the transformed close price.


raw_data = pd.read_csv('../../data/msft_stat.csv', index_col = 0)


# Turn index to DateTime 
raw_data.index = pd.DatetimeIndex(raw_data.index)


raw_data = raw_data.asfreq('D')


# ### Close BoxCox


adf_test(raw_data['Close_boxcox'])


# Although this data in this column is not stationary, I will be working with this column since differencing is automatically applied in the auto arima model.


# ## Cross Validation Train/Test Split 
# ----


tscv = TimeSeriesSplit(n_splits=5)


# ## Auto-ARIMA Forecasting
# -----


from pmdarima import auto_arima



# Setting lam value to lamda value from 03-data-eda
lam = 0.40136288976843615


arima_results_df = pd.DataFrame(columns = ['MSE', 'MAE', 'RMSE', 'MAPE'], index = [1,2,3,4,5])


for i, (train_index, test_index) in enumerate(tscv.split(raw_data)):
    
    # Create train/test datraframes using the indexes from tcsv
    train_df = raw_data.iloc[train_index, :]
    test_df = raw_data.iloc[test_index, :]

    # define y 
    y = train_df['Close_boxcox']

    # Auto-ARIMA auto fits the model 
    arima_model = auto_arima(y, seasonal=False, stepwise=True)

    pred_y = arima_model.predict(n_periods=test_df.shape[0])
    pred_y = inv_boxcox(pred_y, lam)

    # Plotting the forecasted values alongside actual data and previous data (training data) 
    print(f"Fold {i+1}:") 

    plt_forecast(pred_y, fc_method='ARIMA', train=train_df, test= test_df)
    # Printing model summary for greater clarity on coefficients
    print(arima_model.summary())
    
    results_dict = fcast_evaluation(pred_y.values, test_df['Close'].values)
    arima_results_df.iloc[i] = results_dict


arima_results_df


# ----
# **Comment:**
# 
# Fold 5 represents the best performance among all folds in terms of evalutaion metrics(see above), so only be looking into model summary for fold 5.
# 
# **Model Parameters:**
# 
# - p (Autoregressive Term): 0, model does not use past values in the series to forecast future values, main focus of model appears to be recent errors
# - d (Differencing Order): 1, one level of differencing needed to make data stationary
# - q (Moving Average Term): 1, model uses influence of previous day's error to adjust forecasted values
# 
# **Moving Average L1 Coefficient:** -0.0616, indicates a slight overestimate and so the model adjusts predictions downwards slightly to compensate.
# 


# ## Auto-SARIMA Forecasting
# ------


sarima_results_df = pd.DataFrame(columns = ['MSE', 'MAE', 'RMSE', 'MAPE'], index = [1,2,3,4,5])


for i, (train_index, test_index) in enumerate(tscv.split(raw_data)):
    
    train_df = raw_data.iloc[train_index, :]
    test_df = raw_data.iloc[test_index, :]

    y = train_df['Close_boxcox']

    sarima_model = auto_arima(y, seasonal=True, m=7, stepwise=True)

    pred_y = sarima_model.predict(n_periods=test_df.shape[0])
    pred_y = inv_boxcox(pred_y, lam)
    
    print(f"Fold {i+1}:") 
    plt_forecast(pred_y, fc_method='SARIMA', train=train_df, test= test_df)
    print(sarima_model.summary())
    
    results_dict = fcast_evaluation(pred_y.values, test_df['Close'].values)
    sarima_results_df.iloc[i] = results_dict



sarima_results_df


# ----
# **Comment:**
# 
# Again, only looking at model summary for fold 5.
# 
# 
# **Non-Seasonal Orders:**
# 
#     - p: 1, model using value of 1 previous day to predict the next value
#     - d: 1, one level of differencing to make data stationary
#     - q: 0, no influence on models previous error
# 
# **Seasonal Orders:**
# 
#     - P: 2, includes 2 previous seasons (a week) to make forecasts
#     - D: 0, no seasonal differencing required, data must already be stationary from non-seasonal differencing
#     - Q: 1, model uses error of previous week prediction to adjusts forecasts
#     - s: 7, seasonlity of 7 days patterns repeat each week
# 
# 
# **Non-Seasonal Coefficients:** 
# 
#     - AR.L1 (Autoregressive term): -0.0541, suggests an increase/decrease in price seen in previous day leads to decrease/increase of forecasted value.
#     - MA.L1 (Moving Average term):  No moving average terms used for the non-seasonal component as q = 0
# 
# **Seasonal Coefficients:** 
# 
#     - AR.S.L7: 0.6219, positive coefficient meaning an increase in the price a week a go leads to an increase in forecasted value 7 days later.
#     - AR.S.L14: -0.0965, means negative effect from two weeks ago, i.e. increase in price two weeks ago leads to a decreases in forecasted value.
#     - MA.S.L7:  -0.5649, indicates an overestimate 7 days ago and so the model adjusts predictions downwards to correct.
# 
# 


# ## Auto-SARIMAX Forecasting
# ----


# ### Adding interest rates
# 
# **NOTE:** 
# 
# Interest rates affect stock prices in several ways. 
# 
# When interest rates are low, it’s cheaper for companies to borrow money, which can lead to more investment and higher profits, often boosting stock prices.
# 
# Higher interest rates make bonds and savings accounts more attractive, leading investors to move money away from stocks, which can lower stock prices.
# 
# By including interest rates as an exogenous variable, the model can account for these effects use them to improve the accuracy of its predictions.


# Using yfinance to get interest rates
import yfinance as yf

treasury_data = yf.download('^IRX', start=raw_data.index.min(), end=raw_data.index.max())


raw_data.index.max()


# ### Checking for missing dates


first_day = raw_data.index.min()
last_day = raw_data.index.max()


# Calculate difference between last and first day
last_day -  first_day


# Clearly missing dates
treasury_data.shape


# Calculate full date range from start to end date
full_index = pd.date_range(start=first_day, end=last_day, freq='D')


# Re-index so there are no more missing dates
treasury_data = treasury_data.reindex(full_index)


full_index.difference(treasury_data.index)


treasury_data.isna().sum()


# Fill in missing data for added dates using interpolation
treasury_data= treasury_data.interpolate(method='linear')


treasury_data.isna().sum()


treasury_data = treasury_data.rename(columns={"Adj Close": "interest_rate"})


# ### Merging with raw_data


# Create new df irt_df to hold raw_data data + interest rate data
# Merging raw_data with interest rate on date index
irt_df = pd.merge(raw_data, treasury_data['interest_rate'], left_index=True, right_index=True)


# Checking for null values 
irt_df.isna().sum()


irt_df.index


# ### Adding 5 day moving average


irt_df['5D_M_Avg'] = irt_df['Close_diff'].rolling(window=5).mean()


irt_df.head()


# ### Checking stationarity of X vars


# #### Interest Rate


# Check if interest data is stationary 
adf_test(irt_df['interest_rate'])


# Apply boxcox to stabalise variance - negative values when using irx so will not do boxcox transform
#irt_df['itr_stat'], _ = boxcox(irt_df['interest_rate'])


# Applying differencing to stabalise the mean 
irt_df['int_r_diff']= irt_df['interest_rate'].diff().dropna()


irt_df = irt_df.dropna()


adf_test(irt_df['int_r_diff'])


# #### 5-Day Moving Average


adf_test(irt_df['5D_M_Avg'])


# #### Volume


adf_test(irt_df['Volume'])


# ### Auto-SARIMAX


sarimax_results_df = pd.DataFrame(columns = ['MSE', 'MAE', 'RMSE', 'MAPE'], index = [1,2,3,4,5])


raw_data


for i, (train_index, test_index) in enumerate(tscv.split(irt_df)):
    
    train_df = irt_df.iloc[train_index, :]
    test_df = irt_df.iloc[test_index, :]

    # define X and y
    y = train_df['Close_boxcox']
    X = train_df[['int_r_diff', 'Volume','5D_M_Avg']]

    sarimax_model = auto_arima(y, exogenous = X,seasonal=True, m=7, stepwise=True)
    future_exog = test_df[['int_r_diff', 'Volume', '5D_M_Avg']]

    pred_y = sarimax_model.predict( exog=future_exog, n_periods=test_df.shape[0])
    pred_y = inv_boxcox(pred_y, lam)

    print(f"Fold {i+1}:") 
    plt_forecast(pred_y, fc_method='SARIMAX', train=train_df, test= test_df)
    print(sarimax_model.summary())
    
    results_dict = fcast_evaluation(pred_y.values, test_df['Close'].values)
    sarimax_results_df.iloc[i] = results_dict



sarimax_results_df


# ----
# **Comment:**
# 
# Again, only looking at model summary for fold 5 as this fold shows the best performance.
# 
# 
# **Non-Seasonal Orders:**
# 
#     - p: 1, no autoregressive terms used in the model
#     - d: 1, one level of differencing to make data stationary
#     - q: 0, model uses influence of previous day's error to adjust forecasted values
# 
# **Seasonal Orders:**
# 
#     - P: 2, includes 2 previous seasons (a week) to make forecasts
#     - D: 0, no seasonal differencing required, data must already be stationary from non-seasonal differencing
#     - Q: 1, model uses error of previous week prediction to adjusts forecasts
#     - s: 7, seasonlity of 7 days patterns repeat each week
# 
# **Non-Seasonal Coefficients:** 
# 
#     - MA.L1 (Moving Average term): -0.0554, indicates a slight adjustment downward in forecasts due to errors from the previous day.
# 
# **Seasonal Coefficients:** 
# 
#     - AR.S.L7: 0.6218 – A positive coefficient suggesting that an increase in the price from one week ago leads to a higher forecasted value.
#     - AR.S.L14: -0.0961 – A negative coefficient indicating that an increase in price two weeks ago results in a decrease in the forecasted value.
#     - MA.S.L7: -0.5653 – A negative coefficient shows that if there was an overestimate 7 days ago, the model adjusts the forecast downward to correct for this.
# 
# **Exogenous Variables:** 
# 
# Since I am using Auto-SARIMAX, I am unable to see the X variables I added in the model summary.
# 
# To inspect these variables, I will have to manually run SARIMAX with the above parameters and take a look.


# ### Manual SARIMAX


from statsmodels.tsa.statespace.sarimax import SARIMAX


# Using train_df, test_df (still set to fold 5 dates) and X and y defined in loop
sarimax_model = SARIMAX(y, exog=X, order=(0, 1, 1), seasonal_order=(2, 0, 1, 7))
result = sarimax_model.fit()


result.summary()



# ----
# **Comment:**
# 
# **Exogenous Variables Coefficients:**
# 
#     - interest rate: -0.1824, showing a strong negative effect on the forecast. Higher interest rates are associated with lower forecasted values, this makes sense since increased interest rates generally slow market activity and potentially lower stock prices.
# 
#     - Volume: -1.589e-09, a very small negative effect on the forecast, volumne has hardly any effect on the forecasted value. This suggests trading volumne does not have a strong impact on the stock price.
# 
#     - 5 Day Moving Avg: 0.5153, shows a strong positive effect on the forecast. A higher 5 day average tends to lead to a substantial increase in the forecasted value, this makes sense as a higher rolling average over the past five days indicates an upward trend in prices, which would lead to higher forecasted values.


# ## Conclusion
# ----
# 
# In this notebook, I have explored the three advanced forecasting mehtods (ARIMA, SARIMA and SARIMAX). Despite an advancement in model complexity, the results show little improvements. This highlights the challenges faced in predicitng stock prices. Stocks are highly volatile and are heavily influenced by events which are unpredictable.
# 
# The volatility observed in the stock data is largely due to occurrence of external events which are impossible to predict such as COVID-19. Unpredictable events led to sharp fluctuations in prices, which made it very difficult to forecast stock prices. 
# 
# Given what I have found so far, in the future I would like to expand this project by:
# 
# 1. Feature Engineering
# 
#     - I performed feature engineering to a degree when thinking of some exogenous variables to include for SARIMAX forecasting. To take this further it may be better to predict a feature which is not Close price. For example, prediciting whether or not a stockholder would see a return on their investment may be a simpler project and make forecasting more manageable. 
# 
# 2. Advanved Models
# 
#     - Explore even more complex methods to create the predictions such as XGBoost or LSTMs 
#     - These models may provide a better forecast but I still think the improvement will be marginal in comparison to the increase in model complexity unless I was to change what I predict.
# 
# Overall, while the current models offer valuable insights, I would like to revisit this project again in the future and take into account some of the above points.


