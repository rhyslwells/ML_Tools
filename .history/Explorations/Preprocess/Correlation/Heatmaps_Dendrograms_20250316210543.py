import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define dataset
country_data = {
    'Country': ['USA', 'Canada', 'France', 'UK', 'Germany', 'Australia'],
    'Latitude': [0.186, 0.7286, 0.2419, 0.4677, 0.3787, -2.0034],
    'Longitude': [-1.0892, -1.0086, 0.1379, 0.0809, 0.2304, 1.6486],
    'Language': [0, 0, 1, 0, 2, 0]
}

# Create DataFrame
data = pd.DataFrame(country_data).set_index('Country')

# Prepare the dataset for clustering
x_scaled = data.drop(columns=['Language'])  # Remove unnecessary variables

# Display the dataset
print(x_scaled)

# Plot the hierarchical cluster map
sns.clustermap(x_scaled, cmap='mako')
plt.show()

# Plot the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(x_scaled.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
