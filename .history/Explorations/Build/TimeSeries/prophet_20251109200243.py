"""
Example: Hyperparameter tuning and forecasting with Prophet.

This script demonstrates:
1. Installing and importing Prophet
2. Using a parameter grid for model tuning
3. Performing cross-validation
4. Selecting the best parameter set
5. Generating forecasts and plotting results
"""

# ----------------------------------------
# Imports
# ----------------------------------------

# Prophet (new namespace)
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import add_changepoints_to_plot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


# ----------------------------------------
# Parameter grid for tuning
# ----------------------------------------
# changepoint_prior_scale controls flexibility of trend
# seasonality_prior_scale controls strength of seasonal components

param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0]
}

# Build list of all parameter combinations
params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]

mses = []

# ----------------------------------------
# Cross-validation setup
# ----------------------------------------
# Cutoffs define the training endpoints used for rolling forecasts.
# Here, yearly cutoffs from 2009–2020.

cutoffs = pd.date_range(
    start='2009-01-31',
    end='2020-01-31',
    freq='12M'
)

# ----------------------------------------
# Hyperparameter search
# ----------------------------------------

for param in params:

    # Initialise model with parameter combination
    m = Prophet(**param)

    # Add holiday effects
    m.add_country_holidays(country_name='US')

    # Fit to training data (train must have columns ['ds','y'])
    m.fit(train)

    # Cross-validation: produces predictions beyond each cutoff
    df_cv = cross_validation(
        model=m,
        horizon='365 days',
        cutoffs=cutoffs
    )

    # Compute forecasting accuracy metrics
    df_p = performance_metrics(df_cv, rolling_window=1)

    # Record the mean squared error (MSE)
    mses.append(df_p['mse'].values[0])

# ----------------------------------------
# Select best parameters
# ----------------------------------------

tuning_results = pd.DataFrame(params)
tuning_results['mse'] = mses

# Lowest MSE
best_params = params[np.argmin(mses)]

print("Best parameters:")
print(best_params)

# ----------------------------------------
# Fit final model and forecast
# ----------------------------------------

# Fit a new model using the best parameters
m = Prophet(**best_params)
m.add_country_holidays(country_name='US')
m.fit(train)

# Create future dataframe for 12 months ahead
future = m.make_future_dataframe(periods=12, freq='M')

# Predict
forecast = m.predict(future)

# ----------------------------------------
# Plotting
# ----------------------------------------
fig, ax = plt.subplots()

# Plot training series
ax.plot(train['y'], label='Train')

# Plot test and baseline
ax.plot(test['y'], label='Actual', color='blue')
ax.plot(test['baseline'], linestyle=':', color='black', label='Baseline')

# Predictions (test['yhat'] must be generated earlier)
ax.plot(test['yhat'], linestyle='--', linewidth=3, color='darkorange', label='Forecast')

ax.set_xlabel('Date')
ax.set_ylabel('Proportion of searches using keyword "chocolate"')

# Highlight region (example range)
ax.axvspan(204, 215, color='#808080', alpha=0.1)

ax.legend(loc='best')

# X-axis ticks, here years mapped to indices
plt.xticks(np.arange(0, 215, 12), np.arange(2004, 2022, 1))

# Confidence interval shading
plt.fill_between(
    x=test.index,
    y1=test['yhat_lower'],
    y2=test['yhat_upper'],
    color='lightblue'
)

plt.xlim(180, 215)

fig.autofmt_xdate()
plt.tight_layout()

plt.savefig('figures/CH19_F15_peixeiro.png', dpi=300)
plt.show()
