import pandas as pd
import numpy as np

# Function to load dataset
def load_data(file_path):
    """Load a dataset from a CSV file."""
    return pd.read_csv(file_path)


# Function to unpack nested data (e.g., "summer,fall") in a column
def unpack_nested_data(df, column):
    """Unpack nested data in a column (comma-separated strings) using explode."""
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
    
    # Step 3: Handle missing data
    df.fillna(np.nan, inplace=True)  # Fill missing data with NaN (if not already)

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


# General function to fill missing values using group-based aggregation method
def fill_missing_with_group_aggregation_method(df, target_column, group_by_column, agg_method='mean'):
    """Fill missing values in a target column with aggregated values (mean, sum, etc.) by group."""    
    # Count missing values
    missing_indices = df[df[target_column].isnull()].index
    missing_count = len(missing_indices)
    print(f"\nNumber of unique indices with missing values in '{target_column}': {missing_count}")
    print(f"\nFilling {missing_count} missing values in '{target_column}' using '{agg_method}' grouped by '{group_by_column}':\n")

    if missing_count == 0:
        print("No missing values detected. No action needed.")
        return df

    fill_summary = {}  # Dictionary to track how many values were filled per group
    
    # If the aggregation method is 'most_frequent', compute mode
    if agg_method == 'most_frequent':
        # Compute the most frequent value (mode) per group
        mode_values = df.groupby(group_by_column)[target_column].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        
        # Fill missing values with the most frequent value per group
        for group_value, mode_value in mode_values.items():
            rows_to_fill = df[df[group_by_column] == group_value][target_column].isnull()
            if rows_to_fill.any():
                df.loc[df[group_by_column] == group_value, target_column] = df.loc[df[group_by_column] == group_value, target_column].fillna(mode_value)
                fill_summary[group_value] = rows_to_fill.sum()

                print(f"Group '{group_value}': Filled {fill_summary[group_value]} missing values with mode '{mode_value}'")

    else:
        # Compute aggregated values (e.g., mean, sum) per group
        agg_func = df.groupby(group_by_column)[target_column].transform(agg_method)

        # Fill missing values with the aggregated value per group
        for group_value, agg_value in agg_func.groupby(df[group_by_column]).first().items():
            rows_to_fill = df[df[group_by_column] == group_value][target_column].isnull()
            if rows_to_fill.any():
                df.loc[df[group_by_column] == group_value, target_column] = df.loc[df[group_by_column] == group_value, target_column].fillna(agg_value)
                fill_summary[group_value] = rows_to_fill.sum()

                print(f"Group '{group_value}': Filled {fill_summary[group_value]} missing values with '{agg_method}' value '{round(agg_value, 2)}'")

    return df

def summarize_preprocessed_data(df):
    print("\nPreprocessed Data Summary:")
    print(df.describe())  # Summary statistics
    print("\nMissing Data After Preprocessing:")
    print(df.isnull().sum())
    print("\nData Types After Preprocessing:")
    print(df.dtypes)


# Main function to manage preprocessing workflow
def main():
    # Set the file path (adjust as needed)
    file_path = '../../Datasets/EDA_Example.csv'
    
    # Load the dataset
    df_unprocessed = load_data(file_path)
    print("\nDataset Preview:", df_unprocessed.head())
    
    # Preprocess data, including exploding and encoding columns like 'season'
    df = preprocess_data(df_unprocessed, columns_to_explode=['season'], columns_to_encode=['season'])
    print("\nDataset Preview After Preprocessing:", df.head())

    # Missing data summary
    print("\nMissing Data Summary for columns with missing data:")
    get_missing_data_summary(df)

    # Fill missing data with aggregation method (mean for total_sales, most frequent for demand_tags)
    df = fill_missing_with_group_aggregation_method(df, 'total_sales', 'region', agg_method='mean')
    df = fill_missing_with_group_aggregation_method(df, 'demand_tags', 'region', agg_method='most_frequent')

    print("\nMissing Data Summary After Filling:")
    get_missing_data_summary(df)

    print("\nFinal Dataset Preview:", df.head())

    summarize_preprocessed_data(df)
if __name__ == "__main__":
    main()
