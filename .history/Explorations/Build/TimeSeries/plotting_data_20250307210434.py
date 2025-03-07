import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import TimeSeriesSplit

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
def explore_dataset(df, name):
    print(f"\n=== {name} ===")
    print(df.describe())
    print(f"Missing values:\n{df.isnull().sum()}")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['value'], mode='lines', name=name))
    fig.update_layout(title=f"Time Series Plot: {name}", xaxis_title="Timestamp", yaxis_title="Value")
    fig.show()

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
example_df = datasets["art_daily_flatmiddle.csv"]
decomposition = seasonal_decompose(example_df['value'], period=1440)  # Assuming daily periodicity

decomposition.plot()
