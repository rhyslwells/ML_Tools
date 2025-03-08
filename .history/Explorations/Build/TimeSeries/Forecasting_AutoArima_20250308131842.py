import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf
import warnings

warnings.simplefilter("ignore", category=UserWarning)

# Helper functions
def adf_test(series):
    """Perform Augmented Dickey-Fuller test for stationarity."""
    result = adfuller(series.dropna())
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Critical Values: ')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

def plot_acf_pacf(series, lags=40):
    """Plot ACF and PACF."""
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    sm.graphics.tsa.plot_acf(series, lags=lags, ax=ax[0])
    sm.graphics.tsa.plot_pacf(series, lags=lags, ax=ax[1])
    plt.show()

def plt_forecast(pred_y_arima, pred_y_sarima, fc_method_arima, fc_method_sarima, train, test):
    """Plot both ARIMA and SARIMA forecast results."""
    plt.figure(figsize=(10, 6))
    plt.plot(train.index, train['value'], label='Train')
    plt.plot(test.index, test['value'], label='Test')
    plt.plot(test.index, pred_y_arima, label=f'{fc_method_arima} Forecast', linestyle='--')
    plt.plot(test.index, pred_y_sarima, label=f'{fc_method_sarima} Forecast', linestyle='-.')

    plt.title(f"ARIMA vs SARIMA Forecast Comparison")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

def fcast_evaluation(pred, true):
    """
    Evaluate forecasting model performance.
    Ensure that both predicted and true values are numpy arrays, and handle NaNs.
    """
    pred = np.array(pred, dtype=float)  # Ensure numerical type
    true = np.array(true, dtype=float)  # Ensure numerical type

    # Flatten 'true' if it's 2D (e.g., shape (243, 1))
    if true.ndim > 1:
        true = true.flatten()

    # Handle NaN values in both pred and true by forward and backward filling
    pred = pd.Series(pred).ffill().bfill().values  # Forward and backward fill NaN values in predictions
    true = pd.Series(true).ffill().bfill().values  # Forward and backward fill NaN values in actual values

    # Check for any remaining NaN values after forward filling
    if np.any(np.isnan(pred)) or np.any(np.isnan(true)):
        raise ValueError("Predicted or actual values contain NaN after forward filling.")

    # Check if lengths match
    if len(pred) != len(true):
        raise ValueError("Predicted and actual values must have the same length")

    # Calculate metrics
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
raw_data = data.copy()
raw_data = raw_data.asfreq('D')  # Ensure daily frequency

# Optional: Box-Cox Transformation
lam = 0.40136288976843615
raw_data['Close_boxcox'] = boxcox(raw_data['value'], lam)

# Perform ADF test for stationarity
adf_test(raw_data['Close_boxcox'])

# Plot ACF and PACF for better parameter selection
plot_acf_pacf(raw_data['Close_boxcox'])

# Train/Test Split (80/20 split as an example)
train_size = int(len(raw_data) * 0.8)
train_df, test_df = raw_data[:train_size], raw_data[train_size:]

# Function to perform Grid Search for ARIMA parameters
def grid_search_arima(y, p_values, d_values, q_values):
    best_aic = float('inf')
    best_params = None
    best_model = None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = sm.tsa.ARIMA(y, order=(p, d, q))
                    model_fitted = model.fit()
                    if model_fitted.aic < best_aic:
                        best_aic = model_fitted.aic
                        best_params = (p, d, q)
                        best_model = model_fitted
                except:
                    continue
    return best_model, best_params

# Grid search for ARIMA parameters
p_values = range(0, 3)
d_values = range(0, 2)
q_values = range(0, 3)

y = train_df['Close_boxcox']
best_arima_model, best_arima_params = grid_search_arima(y, p_values, d_values, q_values)

# Make predictions using the best ARIMA model
pred_y_arima = inv_boxcox(best_arima_model.forecast(len(test_df)), lam)

def grid_search_sarima_reduced(y, p_values, d_values, q_values, P_values, D_values, Q_values, seasonal_periods):
    best_aic = float('inf')
    best_params = None
    best_model = None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                for P in P_values:
                    for D in D_values:
                        for Q in Q_values:
                            for s in seasonal_periods:
                                try:
                                    model = sm.tsa.SARIMAX(y, order=(p, d, q), seasonal_order=(P, D, Q, s))
                                    model_fitted = model.fit()
                                    if model_fitted.aic < best_aic:
                                        best_aic = model_fitted.aic
                                        best_params = (p, d, q, P, D, Q, s)
                                        best_model = model_fitted
                                except:
                                    continue
    return best_model, best_params

# Reduced parameter ranges
p_values = range(0, 2)  # Testing only 0 and 1 for non-seasonal autoregressive terms
d_values = range(0, 2)  # Testing only 0 and 1 for differencing
q_values = range(0, 2)  # Testing only 0 and 1 for non-seasonal moving average terms

P_values = range(0, 2)  # Testing only 0 and 1 for seasonal autoregressive terms
D_values = range(0, 2)  # Testing only 0 and 1 for seasonal differencing
Q_values = range(0, 2)  # Testing only 0 and 1 for seasonal moving average terms

seasonal_periods = [12]  # Assuming monthly data, seasonality could be yearly (12 months)

best_sarima_model, best_sarima_params = grid_search_sarima_reduced(y, p_values, d_values, q_values, P_values, D_values, Q_values, seasonal_periods)

print("Best SARIMA parameters:", best_sarima_params)



# Make predictions using the best SARIMA model
pred_y_sarima = inv_boxcox(best_sarima_model.forecast(len(test_df)), lam)

# Plot both ARIMA and SARIMA forecasts
plt_forecast(pred_y_arima, pred_y_sarima, fc_method_arima='ARIMA', fc_method_sarima='SARIMA', train=train_df, test=test_df)

# Evaluation (for ARIMA and SARIMA)
arima_metrics = fcast_evaluation(pred_y_arima, test_df['value'].values)
sarima_metrics = fcast_evaluation(pred_y_sarima, test_df['value'].values)

print("Best ARIMA parameters:", best_arima_params)
print("Best SARIMA parameters:", best_sarima_params)
print("ARIMA Evaluation:", arima_metrics)
print("SARIMA Evaluation:", sarima_metrics)
