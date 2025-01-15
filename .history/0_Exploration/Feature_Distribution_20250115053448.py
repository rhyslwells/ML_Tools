import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, binom, poisson, norm, t, chi2, expon, logistic

# Set the style for plots
sns.set(style="whitegrid")

# Create a function to visualize distributions
def explore_distributions():
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    dist_info = []

    # Discrete Distributions
    # 1. Uniform (Discrete)
    data_uniform = np.random.randint(1, 11, size=1000)  # Uniform integers between 1 and 10
    sns.histplot(data_uniform, bins=10, kde=False, ax=axes[0], color="blue")
    axes[0].set_title("Uniform Distribution (Discrete)")
    dist_info.append(("Uniform (Discrete)", pd.Series(data_uniform).describe()))

    # 2. Bernoulli
    data_bernoulli = bernoulli.rvs(p=0.5, size=1000)  # Success probability 0.5
    sns.histplot(data_bernoulli, kde=False, ax=axes[1], color="orange")
    axes[1].set_title("Bernoulli Distribution")
    dist_info.append(("Bernoulli", pd.Series(data_bernoulli).describe()))

    # 3. Binomial
    data_binomial = binom.rvs(n=10, p=0.5, size=1000)  # 10 trials, success probability 0.5
    sns.histplot(data_binomial, bins=10, kde=False, ax=axes[2], color="green")
    axes[2].set_title("Binomial Distribution")
    dist_info.append(("Binomial", pd.Series(data_binomial).describe()))

    # 4. Poisson
    data_poisson = poisson.rvs(mu=3, size=1000)  # Mean (mu) = 3
    sns.histplot(data_poisson, bins=10, kde=False, ax=axes[3], color="purple")
    axes[3].set_title("Poisson Distribution")
    dist_info.append(("Poisson", pd.Series(data_poisson).describe()))

    # Continuous Distributions
    # 5. Gaussian (Normal)
    data_gaussian = norm.rvs(loc=0, scale=1, size=1000)  # Mean=0, StdDev=1
    sns.histplot(data_gaussian, kde=True, ax=axes[4], color="red")
    axes[4].set_title("Gaussian Distribution")
    dist_info.append(("Gaussian", pd.Series(data_gaussian).describe()))

    # 6. T Distribution
    data_t = t.rvs(df=10, size=1000)  # 10 degrees of freedom
    sns.histplot(data_t, kde=True, ax=axes[5], color="cyan")
    axes[5].set_title("T Distribution")
    dist_info.append(("T", pd.Series(data_t).describe()))

    # 7. Chi-squared
    data_chi2 = chi2.rvs(df=4, size=1000)  # 4 degrees of freedom
    sns.histplot(data_chi2, kde=True, ax=axes[6], color="brown")
    axes[6].set_title("Chi-squared Distribution")
    dist_info.append(("Chi-squared", pd.Series(data_chi2).describe()))

    # 8. Exponential
    data_exponential = expon.rvs(scale=1, size=1000)  # Scale = 1
    sns.histplot(data_exponential, kde=True, ax=axes[7], color="pink")
    axes[7].set_title("Exponential Distribution")
    dist_info.append(("Exponential", pd.Series(data_exponential).describe()))

    # 9. Logistic
    data_logistic = logistic.rvs(loc=0, scale=1, size=1000)  # Mean=0, Scale=1
    sns.histplot(data_logistic, kde=True, ax=axes[8], color="gray")
    axes[8].set_title("Logistic Distribution")
    dist_info.append(("Logistic", pd.Series(data_logistic).describe()))

    # Adjust layout and show
    plt.tight_layout()
    plt.show()

    # Summarize and print distribution statistics
    for name, stats in dist_info:
        print(f"\n{name} Summary:\n{stats}")

# Call the function to explore distributions
explore_distributions()
