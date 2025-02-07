import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, binom, poisson, norm, t, chi2, expon, logistic
from sklearn.preprocessing import StandardScaler

# Set the style for plots
sns.set(style="whitegrid")

# Function to explore distributions with standardized features and box plot grouping by distribution nature
def explore_distributions_by_group():
    # Generate synthetic dataset with categorical and numerical features
    np.random.seed(42)
    df = pd.DataFrame({
        'Survived': np.random.choice(['Yes', 'No'], size=1000, p=[0.5, 0.5]),  # Categorical
        'Category': np.random.choice(['A', 'B', 'C'], size=1000, p=[0.3, 0.4, 0.3]),  # Another Categorical
    })

    # Add numerical features for each distribution type
    distributions = {
        'Uniform_Discrete': np.random.randint(1, 11, size=1000),
        'Bernoulli': bernoulli.rvs(p=0.5, size=1000),
        'Binomial': binom.rvs(n=10, p=0.5, size=1000),
        'Poisson': poisson.rvs(mu=3, size=1000),
        'Gaussian': norm.rvs(loc=0, scale=1, size=1000),
        'T_Distribution': t.rvs(df=10, size=1000),
        'Chi_squared': chi2.rvs(df=4, size=1000),
        'Exponential': expon.rvs(scale=1, size=1000),
        'Logistic': logistic.rvs(loc=0, scale=1, size=1000),
    }

    df = pd.concat([df, pd.DataFrame(distributions)], axis=1)

    # -------------------
    # Standardize Features (for symmetric distributions)
    # -------------------
    symmetric_features = ['Gaussian', 'T_Distribution', 'Logistic']
    skewed_features = ['Chi_squared', 'Exponential']

    scaler = StandardScaler()
    standardized_symmetric = scaler.fit_transform(df[symmetric_features])
    standardized_df = pd.DataFrame(standardized_symmetric, columns=[f"{col}_Standardized" for col in symmetric_features])
    df = pd.concat([df, standardized_df], axis=1)

    # -------------------
    # Box Plot Comparison by Distribution Type
    # -------------------

    # Skewed Features
    print("Box Plot for Skewed Distributions...\n")
    melted_skewed = df.melt(id_vars=['Survived'], value_vars=skewed_features, 
                            var_name='Feature', value_name='Value')
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Feature', y='Value', hue='Survived', data=melted_skewed, palette='muted')
    plt.title('Box Plot of Skewed Features (Chi-squared & Exponential)')
    plt.xlabel('Feature')
    plt.ylabel('Value')
    plt.legend(title='Survived')
    plt.tight_layout()
    plt.show()

    # Symmetric Features
    print("Box Plot for Symmetric Distributions...\n")
    melted_symmetric = df.melt(id_vars=['Survived'], value_vars=[f"{col}_Standardized" for col in symmetric_features], 
                               var_name='Feature', value_name='Value')
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Feature', y='Value', hue='Survived', data=melted_symmetric, palette='muted')
    plt.title('Box Plot of Standardized Symmetric Features')
    plt.xlabel('Feature')
    plt.ylabel('Standardized Value')
    plt.legend(title='Survived')
    plt.tight_layout()
    plt.show()

    # -------------------
    # Categorical Feature Distribution
    # -------------------
    print("Exploring Categorical Features...\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.countplot(x='Survived', data=df, ax=axes[0], palette="pastel")
    axes[0].set_title('Distribution of Survived (Yes/No)')
    axes[0].set_ylabel('Count')
    sns.countplot(x='Category', data=df, ax=axes[1], palette="muted")
    axes[1].set_title('Distribution of Categories (A/B/C)')
    axes[1].set_ylabel('Count')
    plt.tight_layout()
    plt.show()

    # -------------------
    # Numerical Feature Distribution
    # -------------------
    print("Exploring Numerical Features...\n")
    features = list(distributions.keys())

    for feature in features:
        print(f"Analyzing {feature} Distribution...")
        g = sns.FacetGrid(df, col="Survived", height=4, aspect=1.2)
        g.map(plt.hist, feature, bins=20, color="skyblue", edgecolor="black")
        g.set_titles(f"{feature} by Survived ({{col_name}})")
        g.set_axis_labels(f"{feature} Value", "Frequency")

        # Add additional statistics to the plot
        for ax, survived_status in zip(g.axes.flat, df['Survived'].unique()):
            feature_data = df[df['Survived'] == survived_status][feature]
            mean_val = feature_data.mean()
            median_val = feature_data.median()
            ax.axvline(mean_val, color='red', linestyle='--', label='Mean')
            ax.axvline(median_val, color='green', linestyle=':', label='Median')
            ax.legend()

        plt.show()

# Call the function to explore distributions
explore_distributions_by_group()
