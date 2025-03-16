import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def pca_anomalies(data, threshold=3.5):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data should be a Pandas DataFrame.")

    print("\nInitial DataFrame Overview:")
    print(data.info())  # Display dataset structure

    # Drop non-numeric columns
    numeric_data = data.select_dtypes(include=[np.number]).copy()
    
    # Handle missing values
    numeric_data.fillna(numeric_data.mean(), inplace=True)
    
    print("\nDataset after selecting numeric columns and filling missing values:")
    print(numeric_data.describe())  # Show summary statistics

    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numeric_data)

    print("\nFirst 5 rows of standardized data:")
    print(pd.DataFrame(scaled_features, columns=numeric_data.columns).head())

    # PCA transformation
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)

    print("\nExplained Variance Ratio (how much variance each component captures):")
    print(pca.explained_variance_ratio_)

    print("\nFirst 5 Principal Components:")
    print(principal_components[:5])

    # Compute reconstruction error
    reconstruction_errors = np.mean((scaled_features - pca.inverse_transform(principal_components)) ** 2, axis=1)

    print("\nFirst 10 Reconstruction Errors:")
    print(reconstruction_errors[:10])

    print("\nMean Reconstruction Error:", np.mean(reconstruction_errors))
    print("Standard Deviation of Reconstruction Error:", np.std(reconstruction_errors))

    # Identify outliers
    potential_outliers = np.where(reconstruction_errors > threshold)[0]
    print(f"\nNumber of Potential Outliers: {len(potential_outliers)}")
    
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
# df = pd.read_csv("../../../Datasets/penguins.csv")
# df=pd.read_csv("../../../Datasets/mtcars.csv")
df=pd.read_csv("../../../Datasets/heart.csv")
# df=pd.read_csv("../../../Datasets/homeprices.csv")


# Apply PCA anomaly detection
clean_df = pca_anomalies(df)

