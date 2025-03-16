import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def pca_anomalies(data, threshold=3.5):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data should be a Pandas DataFrame.")

    # Drop non-numeric columns
    numeric_data = data.select_dtypes(include=[np.number]).copy()

    # Handle missing values
    numeric_data.fillna(numeric_data.mean(), inplace=True)

    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numeric_data)

    # PCA transformation
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)

    # Compute reconstruction error
    reconstruction_errors = np.mean((scaled_features - pca.inverse_transform(principal_components)) ** 2, axis=1)

    # Identify outliers
    potential_outliers = np.where(reconstruction_errors > threshold)[0]

    # Store results
    data = data.copy()  # Avoid modifying original DataFrame
    data['outliers_PCA'] = False
    data.loc[potential_outliers, 'outliers_PCA'] = True

    # Plot PCA results
    plt.figure(figsize=(8, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], c='green', label='Normal Data', alpha=0.5)
    plt.scatter(principal_components[potential_outliers, 0], principal_components[potential_outliers, 1], c='red', label='Potential Outliers')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.title('PCA-Based Anomaly Detection')
    plt.show()

    return data[~data['outliers_PCA']]  # Return only non-outlier data

# Load dataset
df = pd.read_csv("../../../Datasets/penguins.csv")

# Apply PCA anomaly detection
clean_df = pca_anomalies(df)

# Save cleaned data
clean_df.to_csv("penguins_cleaned.csv", index=False)
print("Cleaned DataFrame saved to 'penguins_cleaned.csv'.")

# Example dataset preview
print(clean_df.head())

# Boxplot for a numerical column (bill_length_mm)
plt.figure(figsize=(10, 6))
sns.boxplot(x=clean_df['bill_length_mm'], color='lightcoral')
plt.title('Boxplot for Bill Length')
plt.show()
