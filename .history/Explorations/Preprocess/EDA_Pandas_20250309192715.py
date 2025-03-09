import pandas as pd
import numpy as np
import ast

# Function to load dataset
def load_data(file_path):
    """Load a dataset from a CSV file."""
    return pd.read_csv(file_path)

# Function to detect outliers using IQR method
def detect_outliers(df, column):
    """Detect outliers using Interquartile Range (IQR) method."""
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
        """Safely convert strings to Python literals."""
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []  # Return an empty list if not valid
    
    df[column] = df[column].apply(safe_literal_eval)
    exploded_df = df.explode(column)
    return exploded_df

# General function to handle one-hot encoding for a column with multiple values (like seasons)
def encode_column(df, column):
    """One-hot encode a column containing multiple values (e.g., 'summer, fall')."""
    # Create a set of all possible unique values in the column after exploding
    unique_values = set()
    df[column].dropna().apply(lambda x: unique_values.update(x.split(',')))
    
    # Create dummy columns for each possible season
    for value in unique_values:
        df[value] = df[column].apply(lambda x: 1 if value in x.split(',') else 0)
    
    # Drop the original column after encoding
    df = df.drop(column, axis=1)
    return df

# General preprocessing function
def preprocess_data(df, columns_to_explode=None, columns_to_encode=None):
    """Preprocess the dataset with general steps."""
    if columns_to_explode:
        for column in columns_to_explode:
            df = unpack_nested_data(df, column)
    
    if columns_to_encode:
        for column in columns_to_encode:
            df = encode_column(df, column)
    
    return df

# General function to fill missing values using group-based aggregation method
def fill_missing_with_group_aggregation_method(df, target_column, group_by_column, agg_method='mean'):
    """Fill missing values in a target column with aggregated values (mean, sum, etc.) by group."""
    agg_func = getattr(df.groupby(group_by_column)[target_column], agg_method)()
    df[target_column] = df[target_column].fillna(agg_func)
    return df



# Simplify aggregations using .agg()
def aggregate_data(df, group_by_column, agg_column):
    """Simplify aggregations using .agg() method."""
    summary = df.groupby(group_by_column)[agg_column].agg(['mean', 'sum', 'max'])
    return summary

# Analyze relationships using crosstab
def analyze_relationships(df, column1, column2):
    """Create a crosstab between two columns."""
    return pd.crosstab(df[column1], df[column2], margins=True)

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

# Summarize dataset information
def summarize_dataset(df):
    """Summarize dataset information."""
    print(df.info())  # Get the data types and non-null counts
    for column in df.columns:
        if df[column].dtype == 'object' and not isinstance(df[column].iloc[0], list):  # Avoid list-type columns
            print(f"\nValue counts for column '{column}':")
            print(df[column].value_counts())
    return df.head()

# Main function to manage the workflow
def main():
    begin = "../.."
    path = '/Datasets/EDA_Example.csv'
    file_path = begin + path
    df = load_data(file_path)

    # Preprocessing: General steps before specific dataset preprocessing
    print("\nStarting General Preprocessing...")
    df = preprocess_data(df, columns_to_explode=['season'])

    # Dataset summary
    print("\nDataset Summary:")
    summarize_dataset(df)

    # Missing data summary
    print("\nMissing Data Summary:")
    get_missing_data_summary(df)

    # Fill missing data by group aggregation (mean)
    print("\nFilling Missing Data...")
    df = fill_missing_with_group_aggregation_method(df, 'total_sales', 'region', agg_method='mean')
    print("\nMissing Data Summary After Filling:")
    get_missing_data_summary(df)

    # Detect outliers using IQR method for each numerical column
    print("\nOutliers Detection:")
    for col in df.select_dtypes(include=[np.number]).columns:
        outliers, _ = detect_outliers(df, col)
        print(f"{col} has {outliers.shape[0]} outliers.")

    # Aggregated data by demand_tags
    print("\nAggregated Data by demand_tags:")
    aggregated_data = aggregate_data(df, 'demand_tags', 'total_sales')
    print(aggregated_data)

    # Aggregated data by region
    print("\nAggregated Data by Region:")
    aggregated_data = aggregate_data(df, 'region', 'total_sales')
    print(aggregated_data)

    # Analyze relationships using crosstab
    print("\nCrosstab Analysis:")
    crosstab_result = analyze_relationships(df, 'gender', 'category')
    print(crosstab_result)

if __name__ == "__main__":
    main()
