# %% [markdown]
# <h2 align='center' style='color:blue'>Outlier Detection and Removal Using IQR</h2>

# %%
import pandas as pd
df = pd.read_csv("heights.csv")
df

# %%
df.describe()

# %% [markdown]
# <h3 style='color:purple'>Detect outliers using IQR<h3>

# %%
Q1 = df.height.quantile(0.25)
Q3 = df.height.quantile(0.75)
Q1, Q3

# %%
IQR = Q3 - Q1
IQR

# %%
lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR
lower_limit, upper_limit

# %% [markdown]
# **Here are the outliers**

# %%
df[(df.height<lower_limit)|(df.height>upper_limit)]

# %% [markdown]
# <h3 style='color:purple'>Remove outliers<h3>

# %%
df_no_outlier = df[(df.height>lower_limit)&(df.height<upper_limit)]
df_no_outlier

# %% [markdown]
# <h3 style='color:purple'>Exercise<h3>

# %% [markdown]
# You are given height_weight.csv file which contains heights and weights of 1000 people. Dataset is taken from here,
# https://www.kaggle.com/mustafaali96/weight-height
# 
# You need to do this,
# 
# (1) Load this csv in pandas dataframe and first plot histograms for height and weight parameters
# 
# (2) Using IQR detect weight outliers and print them
# 
# (3) Using IQR, detect height outliers and print them

# %% [markdown]
# [Solution](https://github.com/codebasics/py/tree/master/ML/FeatureEngineering/3_outlier_IQR/Exercise/3_outlier_iqr_exercise.ipynb)


