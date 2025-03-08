import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Configure plotting style
plt.rcParams['figure.figsize'] = (10, 6)

# Load heights dataset
base_path = "../../../Datasets/Outliers/z_score"
df = pd.read_csv(f"{base_path}/heights.csv")

# Display sample data
# print(df.sample(5))

# # Plot histogram of heights
# def plot_histogram(data, xlabel='Height (inches)', ylabel='Count'):
#     plt.hist(data, bins=20, rwidth=0.8, density=True, alpha=0.6, color='b')
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.show()

# # plot_histogram(df.height)

# Plot bell curve along with histogram
plt.hist(df.height, bins=20, rwidth=0.8, density=True, alpha=0.6, color='b')
rng = np.arange(df.height.min(), df.height.max(), 0.1)
plt.plot(rng, norm.pdf(rng, df.height.mean(), df.height.std()), color='r')
plt.xlabel('Height (inches)')
plt.ylabel('Density')
plt.show()

# Compute mean and standard deviation
mean_height = df.height.mean()
std_dev_height = df.height.std()
print(f"Mean height: {mean_height}, Standard Deviation: {std_dev_height}")

# Outlier detection using 3 standard deviations
upper_limit = mean_height + 3 * std_dev_height
lower_limit = mean_height - 3 * std_dev_height
outliers_std = df[(df.height > upper_limit) | (df.height < lower_limit)]
print(f"Outliers detected using standard deviation: {outliers_std.shape}")
print(outliers_std)

# Remove outliers
filtered_df_std = df[(df.height < upper_limit) & (df.height > lower_limit)]
print(f"Shape after removing outliers: {filtered_df_std.shape}")

# Outlier detection using Z-score
df['zscore'] = (df.height - mean_height) / std_dev_height
outliers_z = df[(df.zscore > 3) | (df.zscore < -3)]
print("Outliers detected using Z-score:")
print(outliers_z)

# Remove outliers using Z-score
filtered_df_z = df[(df.zscore > -3) & (df.zscore < 3)]
print(f"Shape after removing Z-score outliers: {filtered_df_z.shape}")
