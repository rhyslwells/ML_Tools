import pandas as pd
import numpy as np

# Sample DataFrame
data = {
    'Category': ['A', 'A', 'B', 'B', 'C', 'C','B'],
    'Value': [10, np.nan, 20, 25, np.nan, 30,np.nan]
}
df = pd.DataFrame(data)

# Display the original DataFrame
print("Original DataFrame:")
print(df)

# Group by 'Category' and calculate the mean for each group
grouped_means = df.groupby('Category')['Value'].transform('mean')
grouped_means

# Fill missing values with the group mean
df['Value'] = df['Value'].fillna(grouped_means)

# Display the DataFrame after filling missing values
print("\nDataFrame after filling missing values with group means:")
print(df)