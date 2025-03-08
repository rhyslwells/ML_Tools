# %%
import pandas as pd

# %%
df = pd.read_csv("..\..\..\Datasets\Outliers\percentile\heights.csv")
df.head()

# %% [markdown]
# <h3 style='color:purple'>Detect outliers using percentile</h3>

# %%
max_thresold = df['height'].quantile(0.95)
max_thresold

# %%
df[df['height']>max_thresold]

# %%
min_thresold = df['height'].quantile(0.05)
min_thresold

# %%
df[df['height']<min_thresold]

# %% [markdown]
# <h3 style='color:purple'>Remove outliers</h3>

# %%
df[(df['height']<max_thresold) & (df['height']>min_thresold)]

# %% [markdown]
# <h3 style='color:purple'>Now lets explore banglore property prices dataset</h3>

# %%
df = pd.read_csv("bhp.csv")
df.head()

# %%
df.shape

# %%
df.describe()

# %% [markdown]
# **Explore samples that are above 99.90% percentile and below 1% percentile rank**

# %%
min_thresold, max_thresold = df.price_per_sqft.quantile([0.001, 0.999])
min_thresold, max_thresold

# %%
df[df.price_per_sqft < min_thresold]

# %%
df[df.price_per_sqft > max_thresold]

# %% [markdown]
# <h3 style='color:purple'>Remove outliers</h3>

# %%
df2 = df[(df.price_per_sqft<max_thresold) & (df.price_per_sqft>min_thresold)]
df2.shape

# %%
df2.describe()

# %% [markdown]
# <h3 style='color:purple'>Exercise</h3>

# %% [markdown]
# Use this air bnb new york city [data set](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data/data) and remove outliers using percentile based on price per night for a given apartment/home. You can use suitable upper and lower limits on percentile based on your intuition. 
# Your goal is to come up with new pandas dataframe that doesn't have the outliers present in it.

# %% [markdown]
# [Solution](https://github.com/codebasics/py/tree/master/ML/FeatureEngineering/1_outliers/Exercise/1_outliers_percentile_exercise.ipynb)


