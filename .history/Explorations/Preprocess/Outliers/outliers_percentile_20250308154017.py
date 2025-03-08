# %%
I want to plot and see the outliers

import pandas as pd

# %%

base="..\..\..\Datasets\Outliers\percentile\"
df = pd.read_csv(base+"heights.csv")
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
df = pd.read_csv(base+"bhp.csv")
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
