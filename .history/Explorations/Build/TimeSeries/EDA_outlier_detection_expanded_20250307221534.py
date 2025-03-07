import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from scipy.stats import zscore
from statsmodels.tsa.arima.model import ARIMA
import ruptures as rpt

# Function to display basic information about the dataset
def display_basic_info(df, name):
    print(f"Dataset: {name}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {', '.join(df.columns)}")
    print(f"Data Types:\n{df.dtypes}")
    print(f"Summary Statistics:\n{df.describe()}\n")
    
# Function to display missing values in the dataset
def display_missing_values(df):
    missing_data = df.isnull().sum()
    missing_data_percent = (missing_data / len(df)) * 100
    missing_info = pd.DataFrame({'Missing Values': missing_data, 'Percentage': missing_data_percent})
    print("Missing Values:\n", missing_info[missing_info['Missing Values'] > 0].sort_values(by='Percentage', ascending=False))


# Function for ARIMA-based anomaly detection
def detect_outliers_arima(df, order=(1, 1, 1)):
    model = ARIMA(df['value'], order=order)
    model_fit = model.fit()
    residuals = model_fit.resid
    z_scores = zscore(residuals)
    outliers_arima = df[np.abs(z_scores) > 3]
    print(f"Potential Outliers (ARIMA Residuals): {len(outliers_arima)} detected")
    return outliers_arima

# Function for IQR-based anomaly detection
def detect_outliers_iqr(df):
    Q1 = df['value'].quantile(0.25)
    Q3 = df['value'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_iqr = df[(df['value'] < lower_bound) | (df['value'] > upper_bound)]
    print(f"Potential Outliers (IQR): {len(outliers_iqr)} detected")
    return outliers_iqr

# Function for Z-score-based anomaly detection
def detect_outliers_z(df, threshold=3.0):
    z_scores = zscore(df['value'])
    outliers_z = df[np.abs(z_scores) > threshold]
    print(f"Potential Outliers (Z-score > {threshold}): {len(outliers_z)} detected")
    return outliers_z

# Function for STL-based anomaly detection
def detect_outliers_stl(df, window=50):
    stl = seasonal_decompose(df['value'], model='additive', period=window)
    residuals = stl.resid.dropna()
    z_scores = zscore(residuals)
    outliers_stl = df.iloc[residuals.index][np.abs(z_scores) > 3]
    print(f"Potential Outliers (STL Residuals): {len(outliers_stl)} detected")
    return outliers_stl

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
    # display_basic_info(df, name)
    # display_missing_values(df)
    
    outliers_iqr = detect_outliers_iqr(df)
    outliers_z = detect_outliers_z(df, z_thresh)
    outliers_stl = detect_outliers_stl(df, window)
    outliers_arima = detect_outliers_arima(df)
    change_points = detect_change_points(df)
    
    plot_dataset(df, name, window, outliers_iqr, outliers_z, outliers_stl, outliers_arima, change_points)

def analyze_single_file(file_path):
    try:
        # Load the CSV file and check if the 'timestamp' column exists
        df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')

        # Print column names and check if the dataset is empty
        print(f"Column names: {df.columns}")
        if df.empty:
            raise ValueError(f"Dataset is empty: {file_path}")
        
        # Check if the 'timestamp' column is present (already the index)
        if 'timestamp' not in df.columns and df.index.name != 'timestamp':
            raise ValueError(f"Missing 'timestamp' column in {file_path}")

        # Duplicate the 'timestamp' index column into a new column
        df['timestamp'] = df.index

        # Ensure the dataset has a time frequency set
        df.index.freq = pd.infer_freq(df.index)
        
        # If only one column exists, reshape it into a 2D DataFrame with the 'value' column
        if df.shape[1] == 1:
            df = pd.DataFrame(df.values, columns=['value'], index=df.index)

        print(df.head())  # Show the first few rows to confirm the structure

        # Proceed with the analysis
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
