# Import required libraries
import pandas as pd

# Define base path for datasets
base = r"..\..\..\Datasets\Outliers\IQR"  # Path for dataset (modify as needed)

# Load heights dataset
df = pd.read_csv(base + r"\heights.csv")  # Use raw string for the full file path

# Display the first few rows of the dataset
print("Dataset preview:")
print(df.head())

# Display descriptive statistics of the dataset
print("\nDescriptive statistics of the dataset:")
print(df.describe())

# Detect outliers using IQR

# Calculate the first (Q1) and third (Q3) quartiles
Q1 = df.height.quantile(0.25)
Q3 = df.height.quantile(0.75)
IQR = Q3 - Q1

# Calculate the lower and upper bounds for outliers
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# Print the IQR and outlier detection thresholds
print(f"\nIQR: {IQR}")
print(f"Lower limit: {lower_limit}, Upper limit: {upper_limit}")

# Return outliers based on the thresholds
outliers = df[(df.height < lower_limit) | (df.height > upper_limit)]
print("\nOutliers detected:")
print(outliers)
