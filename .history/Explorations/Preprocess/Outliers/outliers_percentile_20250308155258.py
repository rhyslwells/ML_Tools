# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")

# Define base path for datasets
base = r"..\..\..\Datasets\Outliers\percentile"  # Removed the trailing backslash

# Load heights dataset
df = pd.read_csv(base + r"\heights.csv")  # Use raw string for the full file path
print(df.head())

# Detect outliers using percentile for the 'height' column
max_threshold = df['height'].quantile(0.95)
print(f"\n \n Max threshold: {max_threshold}")

# Get outliers above the max threshold
outliers_above_max = df[df['height'] > max_threshold]
print("Outliers above max threshold:\n",outliers_above_max)

# Get the min threshold
min_threshold = df['height'].quantile(0.05)
print(f"\n \n Min threshold: {min_threshold}")

# Get outliers below the min threshold
outliers_below_min = df[df['height'] < min_threshold]
print("Outliers below min threshold: \n",outliers_below_min)

# Remove outliers
df_cleaned = df[(df['height'] < max_threshold) & (df['height'] > min_threshold)]
# print(df_cleaned)

# Plotting the outliers in 'height' column
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['height'], color='skyblue')
plt.axvline(x=max_threshold, color='r', linestyle='--', label=f'Max Threshold ({max_threshold:.2f})')
plt.axvline(x=min_threshold, color='g', linestyle='--', label=f'Min Threshold ({min_threshold:.2f})')
plt.title('Boxplot for Heights with Outlier Thresholds')
plt.legend()
plt.show()

#---------------------------------------------------------
# Now, let's explore the Bangalore property prices dataset
df_bhp = pd.read_csv(base + r"\bhp.csv")  # Use raw string for the full file path
print(df_bhp.head())

# Check the shape of the dataset
print(f"\nDataset shape: {df_bhp.shape}")

# Get descriptive statistics for the dataset
# print("\nDetails of the dataset: \n", df_bhp.describe())

# Explore samples that are above 99.90% percentile and below 1% percentile rank for 'price_per_sqft'
min_threshold_bhp, max_threshold_bhp = df_bhp.price_per_sqft.quantile([0.001, 0.999])
print(f"\nMin price_per_sqft threshold: {min_threshold_bhp:.2f}, Max price_per_sqft threshold: {max_threshold_bhp:.2f}")

# Get outliers below the min threshold for price_per_sqft
outliers_below_min_bhp = df_bhp[df_bhp.price_per_sqft < min_threshold_bhp]
print("\nOutliers below min threshold: \n", outliers_below_min_bhp)

# Get outliers above the max threshold for price_per_sqft
outliers_above_max_bhp = df_bhp[df_bhp.price_per_sqft > max_threshold_bhp]
print("\n Outliers above max threshold: \n", outliers_above_max_bhp)

# Concatenate both outliers (above max and below min)
outliers_bhp = pd.concat([outliers_below_min_bhp, outliers_above_max_bhp])

# Sort the outliers in descending order based on 'price_per_sqft'
outliers_bhp_sorted = outliers_bhp.sort_values(by='price_per_sqft', ascending=False)

# Display the sorted outliers
print("\n Outliers in descending order based on price_per_sqft:")
print(outliers_bhp_sorted.shape)
print(outliers_bhp_sorted)

# Remove outliers from the dataset
df_bhp_cleaned = df_bhp[(df_bhp.price_per_sqft < max_threshold_bhp) & (df_bhp.price_per_sqft > min_threshold_bhp)]
print(f"\nCleaned dataset shape: {df_bhp_cleaned.shape}")

# Get descriptive statistics for the cleaned dataset
print("\nCleaned dataset details: \n", df_bhp_cleaned.describe())

# Plotting the outliers in 'price_per_sqft' column
plt.figure(figsize=(10, 6))
sns.boxplot(x=df_bhp_cleaned.price_per_sqft, color='lightcoral')
plt.axvline(x=max_threshold_bhp, color='r', linestyle='--', label=f'Max Threshold ({max_threshold_bhp:.2f})')
plt.axvline(x=min_threshold_bhp, color='g', linestyle='--', label=f'Min Threshold ({min_threshold_bhp:.2f})')
plt.title('Boxplot for Property Prices with Outlier Thresholds')
plt.legend()
plt.show()
