import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import acf

# Directory paths
base_dir = "../../../Datasets/NAB-TimeSeries-Data"
categories = ["artificialNoAnomaly", "artificialWithAnomaly"]

# Load datasets dynamically
datasets = {}

for category in categories:
    dir_path = os.path.join(base_dir, category)
    for file in os.listdir(dir_path):
        if file.endswith(".csv"):
            file_path = os.path.join(dir_path, file)
            df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
            datasets[file] = df  # Store in dictionary


# Function for basic EDA
def explore_dataset(df, name, window=50):
    """
    Perform an enhanced exploratory analysis of a time series dataset.

    Args:
    - df (pd.DataFrame): Time series dataframe with 'value' and 'timestamp' as index.
    - name (str): Dataset name.
    - window (int): Rolling mean window size for trend visualization.

    Returns:
    - None (Prints summary & plots)
    """

    print(f"\n=== {name} ===")
    print(f"Shape: {df.shape}")
    print(f"Time Range: {df.index.min()} to {df.index.max()}")
    
    # Data Summary
    print("\nBasic Statistics:")
    print(df.describe())

    # Missing Values
    missing_pct = df.isnull().mean() * 100
    print("\nMissing Values (%):")
    print(missing_pct)

    # Outlier Detection (IQR)
    Q1, Q3 = df['value'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = df[(df['value'] < lower_bound) | (df['value'] > upper_bound)]
    print(f"\nPotential Outliers: {len(outliers)} detected")

    # Seasonality Analysis (Autocorrelation)
    autocorr_vals = acf(df['value'], nlags=min(100, len(df) - 1), fft=True)
    peak_lag = np.argmax(autocorr_vals[1:]) + 1  # Ignore lag=0
    print(f"Strongest Periodicity at lag: {peak_lag}")

    # Interactive Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['value'], mode='lines', name="Raw Data", opacity=0.6))
    fig.add_trace(go.Scatter(x=df.index, y=df['value'].rolling(window=window, min_periods=1).mean(),
                             mode='lines', name=f"Rolling Mean (window={window})", line=dict(color='red')))
    
    fig.update_layout( 
                      xaxis_title="Timestamp", 
                      yaxis_title="Value",
                      legend=dict(x=0, y=1))
    fig.show()

name="art_daily_no_noise.csv"
explore_dataset(df, name)

# # Explore all datasets
# for name, df in datasets.items():
#     explore_dataset(df, name)


# filenames={
#     "artificialNoAnomaly": [
#         "art_daily_no_noise.csv",
#         "art_daily_perfect_square_wave.csv",
#         "art_daily_small_noise.csv",
#         "art_flatline.csv",
#         "art_noisy.csv"
#     ],
#     "artificialWithAnomaly": [
#         "art_daily_flatmiddle.csv",
#         "art_daily_jumpsdown.csv",
#         "art_daily_jumpsup.csv",
#         "art_daily_nojump.csv",
#         "art_increase_spike_density.csv",
#         "art_load_balancer_spikes.csv"
#     ]
# }


# Example: Decomposition of a dataset
example_df = datasets[name]
decomposition = seasonal_decompose(example_df['value'], period=1440)  # Assuming daily periodicity

decomposition.plot()
