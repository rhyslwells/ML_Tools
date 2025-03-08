import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, kstest, t, chi2, expon, logistic

# Configure plotting style
plt.rcParams['figure.figsize'] = (10, 6)
sns.set(style="whitegrid")

def load_dataset(file_path):
    """Load dataset from the specified path."""
    return pd.read_csv(file_path)

def summarize_data(df):
    """Print dataset summary."""
    print("\nDataset Summary:")
    print(df.info())
    print("\nBasic Statistics:")
    print(df.describe(include='all'))

def goodness_of_fit_tests(data):
    """Determine how well data fits different distributions."""
    distributions = {
        "Gaussian": norm,
        "T": t,
        "Chi-squared": chi2,
        "Exponential": expon,
        "Logistic": logistic
    }
    
    results = {}
    for name, dist in distributions.items():
        params = dist.fit(data)
        _, p_value = kstest(data, lambda x: dist.cdf(x, *params))
        results[name] = p_value
    
    best_fit = max(results, key=results.get)
    return best_fit, results

def plot_distributions(df, col=None):
    """Plot distributions for numerical and categorical columns."""
    if col is not None:
        if col in df.select_dtypes(include=[np.number]).columns:
            numerical_cols = [col]
            categorical_cols = []
        elif col in df.select_dtypes(include=['object', 'category']).columns:
            numerical_cols = []
            categorical_cols = [col]
        else:
            print(f"Column '{col}' not found in dataset.")
            return
    else:
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Plot numerical features
    for col in numerical_cols:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True, stat="density", bins=30, label="Data")
        mu, sigma = df[col].mean(), df[col].std()
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        plt.plot(x, norm.pdf(x, mu, sigma), label="Gaussian Fit", linestyle='dashed')
        
        best_fit, test_results = goodness_of_fit_tests(df[col].dropna())
        plt.title(f"Distribution of {col} (Best Fit: {best_fit})")
        plt.legend()
        plt.show()
        
        print(f"Goodness-of-fit results for {col}")
        for dist, p_value in test_results.items(): print(f"{dist}: {p_value}")
    
    # Plot categorical features
    for col in categorical_cols:
        plt.figure()
        sns.countplot(data=df, x=col, order=df[col].value_counts().index)
        plt.title(f"Distribution of {col}")
        plt.xticks(rotation=45)
        plt.show()

if __name__ == "__main__":
    # Define dataset path
    file_path = "../../Datasets/penguins.csv"  # Modify to handle different datasets
    
    # Load dataset
    df = load_dataset(file_path)
    
    # Summarize data
    summarize_data(df)
    
    # Specify column to analyze (set to None to analyze all columns)
    col = 'bill_depth_mm'
    
    # Plot distributions for the specified column or all columns
    plot_distributions(df, col)