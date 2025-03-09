import pandas as pd
import numpy as np
import ast

# Function to load dataset
def load_data(file_path):
    """Load a dataset from a CSV file with error handling and flexible reading options."""
    try:
        # Attempt to load the CSV with additional parameters to handle malformed data
        df = pd.read_csv(
            file_path,
            error_bad_lines=False,     # Skip lines with too many fields
            warn_bad_lines=True,       # Print warnings for bad lines
            quotechar='"',             # Ensure that quotes are handled correctly
            skipinitialspace=True,     # Ignore leading spaces after a delimiter
            dtype=str                  # Force all columns to be read as strings (optional)
        )
        return df
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

# Function to detect outliers using IQR

def detect_outliers(df, column):
        #Detect outliers using Interquartile Range (IQR) method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    outliers = df[(df[column] < Q1 - 1 * IQR) | (df[column] > Q3 + 1.5 * IQR)]
    df_no_outliers = df[~df.index.isin(outliers.index)]  # Exclude rows matching outliers

    return outliers, df_no_outliers

# General function to unpack nested data in a column
def unpack_nested_data(df, column):
    """Unpack nested data in a column (lists) using explode."""
    def safe_literal_eval(val):
        """Safely convert strings to Python literals, and handle errors."""
        try:
            # If it's a valid string representation of a list, convert it
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            # If it's not a valid literal, return an empty list or handle accordingly
            return []

    # Apply safe_literal_eval to the column
    df[column] = df[column].apply(safe_literal_eval)

    # Ensure that all values are lists before exploding
    exploded_df = df.explode(column)
    return exploded_df

def fill_missing_with_region_mean(df, target_column, group_by_column):
    """Fill missing values in a target column with the mean of the target column grouped by another column."""
    # Calculate the mean of the target_column for each group in group_by_column
    means = df.groupby(group_by_column)[target_column].transform('mean')
    
    # Fill missing values in the target_column with the calculated group-wise mean
    df[target_column] = df[target_column].fillna(means)
    return df

# General preprocessing function
def preprocess_data(df, columns_to_explode=None):
    """Preprocess the dataset with general steps."""
    # Unpack nested columns (explode) if any specified
    if columns_to_explode:
        for column in columns_to_explode:
            df = unpack_nested_data(df, column)

    return df

# Function to simplify aggregations using .agg()
def aggregate_data(df, group_by_column, agg_column):
    """Simplify aggregations using .agg() method."""
    summary = df.groupby(group_by_column)[agg_column].agg(['mean', 'sum', 'max'])
    return summary

# Function to analyze relationships using crosstab()
def analyze_relationships(df, column1, column2):
    """Create a crosstab between two columns."""
    return pd.crosstab(df[column1], df[column2], margins=True)

# General function to apply transformations
def apply_transformations(df, column):
    """Apply custom transformations to the dataframe."""
    df = df.pipe(fill_missing, missing_strategy='mean')
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

def summarize_dataset(df):
    """Summarize dataset information."""
    print(df.info())  # Get the data types and non-null counts
    
    # Loop through each column to apply value_counts() for individual columns
    for column in df.columns:
        if df[column].dtype != 'object':  # Skip non-object columns
            continue
        if isinstance(df[column].iloc[0], list):  # Skip list-type columns
            print(f"Skipping value_counts for column '{column}' due to list-type values.")
        else:
            print(f"\n Value counts for column '{column}':")
            print(df[column].value_counts())
    
    return df.head()


if __name__ == "__main__":
    # Load the dataset
    begin = "../.."
    path = '/Datasets/EDA_Example.csv'
    file_path = begin + path
    df = load_data(file_path)

    # Preprocessing: General steps before specific dataset preprocessing
    print("\nStarting General Preprocessing...")
    df = preprocess_data(df, columns_to_explode=['season'])

    # Fill missing values separately
    print("\nFilling Missing Data...")
    df = fill_missing(df, missing_strategy='mean')

    # Display first few rows and dataset summary
    print("Dataset Summary:")
    summarize_dataset(df)

    # Detect outliers using IQR method for each column
    print("\nOutliers Detection:")
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:  # Apply only on numerical columns
            outliers, df_no_outliers = detect_outliers(df, col)
            print(f"{col} has {outliers.shape[0]} outliers.")

    # Example of aggregating data based on demand_tags
    print("\nAggregated Data by demand_tags:")
    aggregated_data = aggregate_data(exploded_df, 'demand_tags', 'total_sales')
    print(aggregated_data)

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
