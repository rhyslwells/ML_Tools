import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from scipy.stats import zscore
from statsmodels.tsa.arima.model import ARIMA
import ruptures as rpt

# Function for ARIMA-based anomaly detection
def detect_outliers_arima(df, order=(1, 1, 1)):
    model = ARIMA(df['value'], order=order)
    model_fit = model.fit()
    residuals = model_fit.resid
    z_scores = zscore(residuals)
    outliers_arima = df[np.abs(z_scores) > 3]
    print(f"Potential Outliers (ARIMA Residuals): {len(outliers_arima)} detected")
    return outliers_arima

# Change Point Detection
def detect_change_points(df):
    # Use a change point detection algorithm (e.g., Pelt search method)
    model = rpt.Pelt(model="linear").fit(df['value'].values)
    change_points = model.predict(pen=10)
    print(f"Change Points Detected: {change_points}")
    return change_points

# Function to plot the dataset with interactive components
def plot_dataset(df, name, window, outliers_iqr, outliers_z, outliers_stl, outliers_arima, change_points):
    fig = go.Figure()

    # Raw Data
    fig.add_trace(go.Scatter(
        x=df.index, y=df['value'], 
        mode='lines', name="Raw Data", 
        opacity=0.6, showlegend=True
    ))

    # Rolling Mean
    fig.add_trace(go.Scatter(
        x=df.index, y=df['value'].rolling(window=window, min_periods=1).mean(),
        mode='lines', name=f"Rolling Mean (window={window})", 
        line=dict(color='red'), showlegend=True
    ))

    # Highlight IQR Outliers
    if not outliers_iqr.empty:
        fig.add_trace(go.Scatter(
            x=outliers_iqr.index, y=outliers_iqr['value'], 
            mode='markers', name=f"IQR Outliers ({len(outliers_iqr)})", 
            marker=dict(color='purple', size=6, symbol='x'), showlegend=True
        ))

    # Highlight Z-score Outliers
    if not outliers_z.empty:
        fig.add_trace(go.Scatter(
            x=outliers_z.index, y=outliers_z['value'], 
            mode='markers', name=f"Z-score Outliers ({len(outliers_z)})", 
            marker=dict(color='blue', size=6, symbol='circle'), showlegend=True
        ))

    # Highlight STL Outliers
    if not outliers_stl.empty:
        fig.add_trace(go.Scatter(
            x=outliers_stl.index, y=df.loc[outliers_stl.index, 'value'], 
            mode='markers', name=f"STL Outliers ({len(outliers_stl)})", 
            marker=dict(color='orange', size=6, symbol='diamond'), showlegend=True
        ))

    # Highlight ARIMA Outliers
    if not outliers_arima.empty:
        fig.add_trace(go.Scatter(
            x=outliers_arima.index, y=outliers_arima['value'], 
            mode='markers', name=f"ARIMA Outliers ({len(outliers_arima)})", 
            marker=dict(color='green', size=6, symbol='cross'), showlegend=True
        ))

    # Mark Change Points
    for cp in change_points:
        fig.add_trace(go.Scatter(
            x=[df.index[cp]], y=[df['value'][cp]], 
            mode='markers', name="Change Point", 
            marker=dict(color='red', size=8, symbol='star'), showlegend=True
        ))

    fig.update_layout(
        title=f"Time Series Analysis: {name}", 
        xaxis_title="Timestamp", 
        yaxis_title="Value",
        legend=dict(
            x=1,  # Position the legend outside the plot
            y=1.3,  # Place it at the top-right corner
            traceorder='normal',
            orientation='v',
            font=dict(size=10),
            bgcolor='rgba(255, 255, 255, 0)', 
            bordercolor='Black', 
            borderwidth=1
        ),
        margin=dict(r=150)  # Adjust the right margin to make space for the legend
    )
    fig.show()

# Main function for dataset exploration
def explore_dataset(df, name, window=50, z_thresh=3.0):
    display_basic_info(df, name)
    display_missing_values(df)
    
    outliers_iqr = detect_outliers_iqr(df)
    outliers_z = detect_outliers_z(df, z_thresh)
    outliers_stl = detect_outliers_stl(df, window)
    outliers_arima = detect_outliers_arima(df)
    change_points = detect_change_points(df)
    
    plot_dataset(df, name, window, outliers_iqr, outliers_z, outliers_stl, outliers_arima, change_points)

# Function to load dataset and run analysis based on user input
def analyze_single_file(file_path):
    try:
        df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
        name = os.path.basename(file_path)
        explore_dataset(df, name)
    except Exception as e:
        print(f"Error loading dataset: {e}")

# Example usage
begin="../../../"
filenames = ["Datasets/NAB-TimeSeries-Data/artificialWithAnomaly/art_daily_jumpsdown.csv"]
file = filenames[0]
file_path = begin + file
analyze_single_file(file_path)
