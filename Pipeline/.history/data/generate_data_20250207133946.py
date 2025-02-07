from sklearn.datasets import make_classification
import pandas as pd

# Generate a dataset with 1000 samples, 10 features, and imbalanced fraud labels
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                           n_redundant=2, weights=[0.95, 0.05], random_state=42)

# Convert to DataFrame
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
df['fraud'] = y  # Target column

# Save to CSV
df.to_csv("fraud_data.csv", index=False)
print("fraud_data.csv created successfully!")
