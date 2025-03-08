"""
Common techniques for Exploratory Data Analysis (EDA) using Pandas.
The code is generalized to work with any dataset and can be easily adapted for different datasets.
"""

import pandas as pd
import numpy as np
import ast

# Function to load dataset
def load_data(file_path):
    """Load a dataset from a CSV file."""
    return pd.read_csv(file_path)

# Function to detect outliers using IQR
def detect_outliers(df, column):
    """Detect outliers using Interquartile Range (IQR) method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    outliers = df[(df[column] < Q1 - 1 * IQR) | (df[column] > Q3 + 1.5 * IQR)]
    df_no_outliers = df[~df.index.isin(outliers.index)]  # Exclude rows matching outliers

    return outliers, df_no_outliers

# Function to unpack nested data using .explode()
def unpack_nested_data(df, column):
    """Unpack nested data in a column (lists) using explode."""
    # Ensure the column contains a list, not a string
    df[column] = df[column].apply(ast.literal_eval)
    exploded_df = df.explode(column)
    return exploded_df

# Function to simplify aggregations using .agg()
def aggregate_data(df, group_by_column, agg_column):
    """Simplify aggregations using .agg() method."""
    summary = df.groupby(group_by_column)[agg_column].agg(['mean', 'sum', 'max'])
    return summary

# Function to streamline transformations using .pipe()
def fill_missing(df, column, strategy='median'):
    """Fill missing values in a column."""
    if strategy == 'median':
        df[column] = df[column].fillna(df[column].median())
    elif strategy == 'mean':
        df[column] = df[column].fillna(df[column].mean())
    return df

def normalize(df, column):
    """Normalize a numerical column."""
    df[f'{column}_normalized'] = (df[column] - df[column].mean()) / df[column].std()
    return df

def add_features(df, column):
    """Add new features based on existing columns."""
    df[f'{column}_category'] = pd.cut(df[column], bins=[0, 500, 1000, float('inf')],
                                      labels=['Low', 'Medium', 'High'])
    return df

# Function to analyze relationships using crosstab()
def analyze_relationships(df, column1, column2):
    """Create a crosstab between two columns."""
    return pd.crosstab(df[column1], df[column2], margins=True)

# General function to apply transformations
def apply_transformations(df, column):
    """Apply custom transformations to the dataframe."""
    df = (df.pipe(fill_missing, column=column)
             .pipe(normalize, column=column)
             .pipe(add_features, column=column))
    return df

# Function to analyze missing data
def get_missing_data_summary(df):
    """Summarize missing data for each column."""
    total = df.shape[0]
    missing_percent = {}
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            null_count = df[col].isnull().sum()
            percent = (null_count / total) * 100
            missing_percent[col] = percent
            print(f"{col}: {null_count} ({round(percent, 3)}%)")
    return missing_percent

# Function to summarize the dataset info
def summarize_dataset(df):
    """Summarize dataset information."""
    print(df.info())  # Get the data types and non-null counts
    return df.head()  # Show the first few rows

# Main execution
if __name__ == "__main__":
    # Load the dataset
    begin="../.."
    path = '/Datasets/enrollment_forecast.csv'
    file_path=begin+path
    df = load_data(file_path)

    # Display first few rows and dataset summary
    print("Dataset Summary:")
    summarize_dataset(df)

    # Example of detecting outliers in a specific column (e.g., 'total_sales')
    print("\nOutliers Detection:")
    outliers, df_no_outliers = detect_outliers(df, 'total_sales')
    print(f"Outliers: {outliers}")

    # Example of unpacking nested data in the 'tags' column
    print("\nUnpacking Nested Data:")
    exploded_df = unpack_nested_data(df, 'tags')
    print(exploded_df.head())

    # Example of aggregating data based on region
    print("\nAggregated Data by Region:")
    aggregated_data = aggregate_data(df, 'region', 'total_sales')
    print(aggregated_data)

    # Apply transformations using pipe
    print("\nApplying Transformations:")
    transformed_df = apply_transformations(df, 'total_sales')
    print(transformed_df.head())

    # Example of analyzing relationships using crosstab
    print("\nCrosstab Analysis:")
    crosstab_result = analyze_relationships(df, 'gender', 'category')
    print(crosstab_result)

    # Get missing data summary
    print("\nMissing Data Summary:")
    missing_data = get_missing_data_summary(df)
