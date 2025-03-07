import pandas as pd
import numpy as np
import ruptures as rpt
import plotly.graph_objects as go

# Create a synthetic time series dataset with multiple change points
np.random.seed(42)
time = np.arange(0, 100)
data = np.concatenate([
    np.random.normal(0, 1, 30),   # First segment
    np.random.normal(5, 1, 30),   # Second segment (change point at index 30)
    np.random.normal(-3, 1, 40)   # Third segment (change point at index 60)
])

# Create a DataFrame
df = pd.DataFrame({'time': time, 'value': data})

# Change Point Detection Function
def detect_change_points(df):
    # Ensure 'value' column is numeric and has no missing values
    df['value'] = pd.to_numeric(df['value'], errors='coerce')  # Convert to numeric, coerce errors to NaN
    df = df.dropna(subset=['value'])  # Drop rows where 'value' is NaN
    
    if df['value'].empty:
        print("Error: After cleaning, 'value' column is empty.")
        return []  # Return empty list if no data left
    
    # Reshape the data to a 2D array (needed by ruptures)
    values_reshaped = df['value'].values.reshape(-1, 1)  # Reshape to 2D array with one column
    
    # Use a change point detection algorithm (e.g., Pelt search method)
    model = rpt.Pelt(model="linear").fit(values_reshaped)  # Pass 2D array
    change_points = model.predict(pen=10)
    
    # Remove last change point if it is equal to the length of the data (out of bounds)
    if change_points[-1] == len(df):
        change_points = change_points[:-1]
    
    return change_points

# Detect change points
change_points = detect_change_points(df)

# Plotting the time series and the detected change points using Plotly
fig = go.Figure()

# Plot the time series data
fig.add_trace(go.Scatter(x=df['time'], y=df['value'], mode='lines', name='Time Series'))

# Add vertical lines at the change points
for cp in change_points:
    # Ensure the change point index is within bounds
    if cp < len(df):
        fig.add_trace(go.Scatter(
            x=[df['time'][cp], df['time'][cp]],
            y=[df['value'].min(), df['value'].max()],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name=f'Change Point {cp}'
        ))

# Layout settings
fig.update_layout(
    title='Change Point Detection',
    xaxis_title='Time',
    yaxis_title='Value',
    showlegend=True,
    template='plotly_dark'
)

# Show the plot
fig.show()
