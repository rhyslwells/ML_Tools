import pandas as pd
import numpy as np
import ast

# Function to load dataset
def load_data(file_path):
    """Load a dataset from a CSV file."""
    return pd.read_csv(file_path)


# Function to unpack nested data (e.g., "summer,fall") in a column

def unpack_nested_data(df, column):
    "Unpack nested data in a column (comma-separated strings) using explode."
    def safe_literal_eval(val):
        """Safely convert strings to Python literals (e.g., lists)."""
        try:
            # If the value is a string of comma-separated items, split into a list
            return val.split(',') if isinstance(val, str) else val
        except (ValueError, SyntaxError):
            return []  # Return an empty list if not valid
    
    # Apply the function to split the values into lists if needed
    df[column] = df[column].apply(safe_literal_eval)
    
    # If the column contains lists, explode the column to separate rows
    if df[column].apply(lambda x: isinstance(x, list)).all():
        exploded_df = df.explode(column)
    else:
        exploded_df = df  # No explosion if data is not in list format
    
    return exploded_df


# Function to handle one-hot encoding for a column with multiple values (e.g., 'summer, fall')
def encode_column(df, column):
    """One-hot encode a column containing multiple values (e.g., 'summer, fall')."""
    
    # Create a set of all possible unique values in the column
    unique_values = set()
    
    # Handle both comma-separated strings and already exploded lists
    df[column].dropna().apply(lambda x: unique_values.update(x.split(',') if isinstance(x, str) else x))
    
    # Create dummy columns for each unique value found
    for value in unique_values:
        df[value] = df[column].apply(lambda x: 1 if value in (x.split(',') if isinstance(x, str) else x) else 0)
    
    # Drop the original column after encoding
    df = df.drop(column, axis=1)
    return df

# General preprocessing function
def preprocess_data(df, columns_to_explode=None, columns_to_encode=None):
    """Preprocess the dataset with general steps."""

    # Preprocessing: General steps before specific dataset preprocessing
    print("\nStarting General Preprocessing...")

    # set id column as index
    df.set_index('id', inplace=True)
    
    # Step 1: Explode columns with nested data (lists)
    if columns_to_explode:
        for column in columns_to_explode:
            df = unpack_nested_data(df, column)
    
    # Step 2: Apply encoding to any specified columns
    if columns_to_encode:
        for column in columns_to_encode:
            df = encode_column(df, column)
    
    return df


# Function to detect outliers using IQR method
def detect_outliers(df, column):
    """Detect outliers using Interquartile Range (IQR) method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    outliers = df[(df[column] < Q1 - 1 * IQR) | (df[column] > Q3 + 1.5 * IQR)]
    df_no_outliers = df[~df.index.isin(outliers.index)]  # Exclude rows matching outliers

    return outliers, df_no_outliers


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

# General function to fill missing values using group-based aggregation method
# General function to fill missing values using group-based aggregation method
def fill_missing_with_group_aggregation_method(df, target_column, group_by_column, agg_method='mean'):
    """Fill missing values in a target column using a specified aggregation method within each group."""
    OutlierCounter = 0

    # Identify all missing values before any modifications
    missing_indices = df[df[target_column].isnull()].index.unique()
    missing_count = len(missing_indices)

    # Print unique indices before filling
    print(f"\nNumber of unique indices with missing values in '{target_column}': {missing_count}")

    print(f"\nFilling {missing_count} missing values in '{target_column}' using '{agg_method}' grouped by '{group_by_column}':\n")

    if missing_count == 0:
        print("No missing values detected. No action needed.")
        return df

    fill_summary = {}  # Dictionary to track how many values were filled per group

    if agg_method == 'most_frequent':
        # Compute the most frequent value per group
        mode_values = df.groupby(group_by_column)[target_column].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

        # Fill missing values based on mode
        for idx in missing_indices:
            OutlierCounter += 1
            group_value = df.at[idx, group_by_column]
            fill_value = mode_values.get(group_value, np.nan)

            if pd.notna(fill_value):
                print(f"Outlier {OutlierCounter}/{missing_count}: Row {idx}: '{target_column}' filled with '{fill_value}' (Group: '{group_value}')")
                df.at[idx, target_column] = fill_value
                fill_summary[group_value] = fill_summary.get(group_value, 0) + 1
            else:
                print(f"Outlier {OutlierCounter}/{missing_count}: Row {idx}: No valid mode found for group '{group_value}', leaving as NaN")

    else:
        # Compute the aggregated values per group
        agg_func = df.groupby(group_by_column)[target_column].transform(agg_method)

        # Ensure missing indices remain consistent
        print(f"\nNumber of unique indices with missing values in '{target_column}' before filling: {missing_count} \n")
        
        for idx in missing_indices:
            OutlierCounter += 1
            group_value = df.at[idx, group_by_column]
            fill_value = agg_func.at[idx]

            if pd.notna(fill_value):
                print(f"Outlier {OutlierCounter}/{missing_count}: Row {idx}: '{target_column}' filled with '{round(fill_value, 2)}' (Group: '{group_value}')")
                df.at[idx, target_column] = fill_value
                fill_summary[group_value] = fill_summary.get(group_value, 0) + 1
            else:
                print(f"Outlier {OutlierCounter}/{missing_count}: Row {idx}: No valid '{agg_method}' aggregation found for group '{group_value}', leaving as NaN")

    # Summary Output
    print("\nSummary of missing value fills per group:")
    for group, count in fill_summary.items():
        print(f"  - Group '{group}': {count} values filled")

    return df

# Summarize dataset information
def summarize_dataset(df,col_list):
    """Summarize dataset information."""
    print(df.info())  # Get the data types and non-null counts
    for column in col_list:
        print(f"\nValue counts for column '{column}':")
        print(df[column].value_counts())
    return df.head()

# Main function to manage the workflow
def main():
    begin = "../.."
    path = '/Datasets/EDA_Example.csv'
    file_path = begin + path
    df_unprocessed = load_data(file_path)

    # Preprocess data, including exploding and encoding columns like 'season'
    df = preprocess_data(df_unprocessed, columns_to_explode=['season'], columns_to_encode=['season'])

    # print("\nDataset Preview:", df.head())

    # Dataset summary
    # print("\nDataset Summary:")
    # summarize_dataset(df,['region', 'gender', 'category', 'demand_tags','priority', 'summer', 'winter', 'spring', 'fall'])

    # Missing data summary
    print("\nMissing Data Summary for columns with missing data:")
    get_missing_data_summary(df)

    # Fill missing data by group aggregation (mean)
    df = fill_missing_with_group_aggregation_method(df, 'total_sales', 'region', agg_method='mean')

    # Fix: for categoricals i want to fill using the most frequent for that grouping
    df = fill_missing_with_group_aggregation_method(df, 'demand_tags', 'region', agg_method='most_frequent')


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
