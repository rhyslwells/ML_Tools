# Forecasting Model Overview

## Overview

This folder contains several notebooks which explore various forecasting models. 

The goal of the project is to predict stock close prices for the next seven days. Each notebook applies different forecasting methods and evaluates their performance on the dataset using cross-validation with `TimeSeriesSplit`.

## Notebooks Overview

### 1. Baseline Forecasting Methods

This notebook implements basic timeseries forecasting models, these simple models serve as a baseline for more advanced methods.

- **Methods Implemented:**
    - **Mean Forecasting:** Uses the average of all past values as the forecast for future periods.
    - **Naive Forecasting:** The last observed value is used as the forecast for all future periods.
    - **Seasonal Naive Forecasting:** Uses the value from the previous seasonal period to forecast the future.
    - **Drift Method:** Predicts future values based on the trend between the first and last observations in the training data.

### 2. Exponential Smoothing Models

This notebook explores the different exponential smoothing techniques:

- **Methods Implemented:**
    - **Simple Exponential Smoothing:** Suitable for forecasting data without trends or seasonality.
    - **Double Exponential Smoothing (Holts Linear Trend):** Accounts for trends in the data by introducing trend components.
    - **Triple Exponential Smoothing (Holt Winters):** Captures both trend and seasonal components in timeseries data.

### 3. Auto ARIMA and SARIMA Models

This notebook applies ARIMA-based models for timeseries forecasting. These models are more advanced and account for both non-seasonal and seasonal components as well as exogenous variables.

- **Methods Implemented:**
    - **ARIMA:** Models the autocorrelations within the data.
    - **SARIMA:** Extends ARIMA by capturing seasonality in the data.
    - **SARIMAX:** Incorporates external variables (e.g. interest rates, volume) into the forecasting model.

## Cross-Validation Strategy

All notebooks use cross-validation based on `TimeSeriesSplit` to ensure proper evaluation of performance with no data leakage. This method ensures that training and test data are split while maintaining the chronological order of the data.

## Results Overview

### 1. Baseline Forecasting Methods

| Method             | MSE          | MAE         | RMSE        | MAPE        |
|--------------------|--------------|-------------|-------------|-------------|
| **Drift**          | 1716.65943   | 32.93316    | 38.244796   | 11.454145   |
| **Naive**          | 2575.061318  | 40.620242   | 46.571646   | 13.068402   |
| **Seasonal Naive** | 2972.172213  | 43.653461   | 49.507926   | 14.027392   |
| **Mean**           | 10176.668043 | 86.896259   | 91.006774   | 27.540473   |

The Drift method seems to be the best performing baseline forecasting method with the lowest average errors. 
The Naive method is better than the Seasonal Naive and Mean methods but still less accurate than Drift.

### 2. Exponential Smoothing Models

The exponential smoothing methods did not perform well with the stock data.

Simple Exponential Smoothing, Holt's Linear Trend and Holt-Winters’ methods showed varying results but overall they all struggled in capturing trends and seasonality effectively.

Advanced forecasting models such as ARIMA may be needed in order to better capture the complexities seen in this data.

### 3. Auto ARIMA and SARIMA Models

Marginal difference in evaluation metrics across all three ARIMA models, this could be due to the high volatility of stock data.

While ARIMA models are more complex and are able to capture more of the variability in the stock data, the unpredictable nature of stocks (due to some external events) makes forecasting challenging. As a result even these complex methods fail to get high accuracy in their forecasts.

