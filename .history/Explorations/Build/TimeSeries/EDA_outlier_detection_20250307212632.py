import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from scipy.stats import zscore

# Function for enhanced EDA with multiple outlier detection methods
def explore_dataset(df, name, window=50, z_thresh=3.0):
    """
    Perform an exploratory analysis of a time series dataset, 
    including multiple outlier detection techniques.

    Args:
    - df (pd.DataFrame): Time series dataframe with 'value' and 'timestamp' as index.
    - name (str): Dataset name.
    - window (int): Rolling mean window size for trend visualization.
    - z_thresh (float): Threshold for Z-score based outlier detection.

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

    # Outlier Detection (IQR Method)
    Q1, Q3 = df['value'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers_iqr = df[(df['value'] < lower_bound) | (df['value'] > upper_bound)]
    print(f"\nPotential Outliers (IQR): {len(outliers_iqr)} detected")

    # Outlier Detection (Z-score)
    df['zscore'] = zscore(df['value'])
    outliers_z = df[np.abs(df['zscore']) > z_thresh]
    print(f"Potential Outliers (Z-score > {z_thresh}): {len(outliers_z)} detected")

    # Outlier Detection (STL Decomposition - Residual Based)
    try:
        decomposition = seasonal_decompose(df['value'], model='additive', period=window)
        residual = decomposition.resid.dropna()
        residual_std = residual.std()
        outliers_stl = residual[np.abs(residual) > 2 * residual_std]
        print(f"Potential Outliers (STL Residuals): {len(outliers_stl)} detected")
    except ValueError:
        outliers_stl = pd.Series(dtype="float64")
        print("STL Decomposition skipped (not enough data points).")

    # Seasonality Analysis (Autocorrelation)
    autocorr_vals = acf(df['value'], nlags=min(100, len(df) - 1), fft=True)
    peak_lag = np.argmax(autocorr_vals[1:]) + 1  # Ignore lag=0
    print(f"Strongest Periodicity at lag: {peak_lag}")

    # Interactive Plot with Outliers Highlighted
    fig = go.Figure()

    # Raw Data
    fig.add_trace(go.Scatter(
        x=df.index, y=df['value'], 
        mode='lines', name="Raw Data", 
        opacity=0.6
    ))

    # Rolling Mean
    fig.add_trace(go.Scatter(
        x=df.index, y=df['value'].rolling(window=window, min_periods=1).mean(),
        mode='lines', name=f"Rolling Mean (window={window})", 
        line=dict(color='red')
    ))

    # Highlight IQR Outliers
    if not outliers_iqr.empty:
        fig.add_trace(go.Scatter(
            x=outliers_iqr.index, y=outliers_iqr['value'], 
            mode='markers', name=f"IQR Outliers ({len(outliers_iqr)})", 
            marker=dict(color='purple', size=6, symbol='x')
        ))

    # Highlight Z-score Outliers
    if not outliers_z.empty:
        fig.add_trace(go.Scatter(
            x=outliers_z.index, y=outliers_z['value'], 
            mode='markers', name=f"Z-score Outliers ({len(outliers_z)})", 
            marker=dict(color='blue', size=6, symbol='circle')
        ))

    # Highlight STL Outliers
    if not outliers_stl.empty:
        fig.add_trace(go.Scatter(
            x=outliers_stl.index, y=df.loc[outliers_stl.index, 'value'], 
            mode='markers', name=f"STL Outliers ({len(outliers_stl)})", 
            marker=dict(color='orange', size=6, symbol='diamond')
        ))

    # Plot Layout
    fig.update_layout(
        title=f"Time Series Analysis: {name}", 
        xaxis_title="Timestamp", 
        yaxis_title="Value",
        legend=dict(x=0, y=1)
    )
    fig.show()


# Function to load dataset and run analysis based on user input
def analyze_single_file(file_path):
    """
    Load the dataset from the given path and perform the analysis.

    Args:
    - file_path (str): Relative or absolute path to the dataset.

    Returns:
    - None
    """

    # Load dataset
    try:
        df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
        # Extract file name from path for display
        name = os.path.basename(file_path)
        explore_dataset(df, name)
    except Exception as e:
        print(f"Error loading dataset: {e}")

# Example: Pasting relative path for a single file analysis
begin="../../../"
filenames = [
        "Datasets/NAB-TimeSeries-Data/artificialNoAnomaly/art_daily_no_noise.csv",
        "Datasets/NAB-TimeSeries-Data/artificialNoAnomaly/art_daily_perfect_square_wave.csv",
        "Datasets/NAB-TimeSeries-Data/artificialNoAnomaly/art_daily_small_noise.csv",
        "Datasets/NAB-TimeSeries-Data/artificialNoAnomaly/art_flatline.csv",
        "Datasets/NAB-TimeSeries-Data/artificialNoAnomaly/art_noisy.csv",
        "Datasets/NAB-TimeSeries-Data/artificialWithAnomaly/art_daily_flatmiddle.csv",
        "Datasets/NAB-TimeSeries-Data/artificialWithAnomaly/art_daily_jumpsdown.csv",
        "Datasets/NAB-TimeSeries-Data/artificialWithAnomaly/art_daily_jumpsup.csv",
        "Datasets/NAB-TimeSeries-Data/artificialWithAnomaly/art_daily_nojump.csv",
        "Datasets/NAB-TimeSeries-Data/artificialWithAnomaly/art_increase_spike_density.csv",
        "Datasets/NAB-TimeSeries-Data/artificialWithAnomaly/art_load_balancer_spikes.csv"
        ]
file=filenames[0]

file_path=begin+file
analyze_single_file(file_path)
