# %% [markdown]
# <h2 align='center' style='color:purple'>Outlier detection and removal using z-score and standard deviation in python pandas</h2>

# %%
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (10,6)

# %% [markdown]
# **We are going to use heights dataset from kaggle.com. Dataset has heights and weights both but I have removed weights to make it simple**

# %% [markdown]
# https://www.kaggle.com/mustafaali96/weight-height

# %%
df = pd.read_csv("heights.csv")
df.sample(5)

# %%
plt.hist(df.height, bins=20, rwidth=0.8)
plt.xlabel('Height (inches)')
plt.ylabel('Count')
plt.show()

# %% [markdown]
# Read this awesome article to get your fundamentals clear on normal distribution, bell curve and standard deviation. https://www.mathsisfun.com/data/standard-normal-distribution.html

# %% [markdown]
# **Plot bell curve along with histogram for our dataset**

# %%
from scipy.stats import norm
import numpy as np
plt.hist(df.height, bins=20, rwidth=0.8, density=True)
plt.xlabel('Height (inches)')
plt.ylabel('Count')

rng = np.arange(df.height.min(), df.height.max(), 0.1)
plt.plot(rng, norm.pdf(rng,df.height.mean(),df.height.std()))

# %%
df.height.mean()

# %%
df.height.std()

# %% [markdown]
# Here the mean is 66.37 and standard deviation is 3.84. 

# %% [markdown]
# <h3 style='color:blue'>(1) Outlier detection and removal using 3 standard deviation</h3>

# %% [markdown]
# One of the ways we can remove outliers is remove any data points that are beyond **3 standard deviation** from mean. Which means we can come up with following upper and lower bounds

# %%
upper_limit = df.height.mean() + 3*df.height.std()
upper_limit

# %%
lower_limit = df.height.mean() -3*df.height.std()
lower_limit

# %% [markdown]
# Here are the outliers that are beyond 3 std dev from mean

# %%
df[(df.height>upper_limit) | (df.height<lower_limit)]

# %% [markdown]
# Above the heights on higher end is **78 inch** which is around **6 ft 6 inch**. Now that is quite unusual height. There are people who have this height but it is very uncommon and it is ok if you remove those data points.
# Similarly on lower end it is **54 inch** which is around **4 ft 6 inch**. While this is also a legitimate height you don't find many people having this height so it is safe to consider both of these cases as outliers

# %% [markdown]
# **Now remove these outliers and generate new dataframe**

# %%
df_no_outlier_std_dev = df[(df.height<upper_limit) & (df.height>lower_limit)]
df_no_outlier_std_dev.head()

# %%
df_no_outlier_std_dev.shape

# %%
df.shape

# %% [markdown]
# Above shows original dataframe data 10000 data points. Out of that we removed 7 outliers (i.e. 10000-9993) 

# %% [markdown]
# <h3 style='color:blue'>(2) Outlier detection and removal using Z Score</h3>

# %% [markdown]
# **Z score is a way to achieve same thing that we did above in part (1)**

# %% [markdown]
# **Z score indicates how many standard deviation away a data point is.**
# 
# For example in our case mean is 66.37 and standard deviation is 3.84. 
# 
# If a value of a data point is 77.91 then Z score for that is 3 because it is 3 standard deviation away (77.91 = 66.37 + 3 * 3.84)

# %% [markdown]
# <h3 style='color:purple'>Calculate the Z Score</h3>

# %% [markdown]
# <img align='left' height="400" width="300" src="zscore.png" />

# %%
df['zscore'] = ( df.height - df.height.mean() ) / df.height.std()
df.head(5)

# %% [markdown]
# Above for first record with height 73.84, z score is 1.94. This means 73.84 is 1.94 standard deviation away from mean

# %%
(73.84-66.37)/3.84

# %% [markdown]
# **Get data points that has z score higher than 3 or lower than -3. Another way of saying same thing is get data points that are more than 3 standard deviation away**

# %%
df[df['zscore']>3]

# %%
df[df['zscore']<-3]

# %% [markdown]
# Here is the list of all outliers 

# %%
df[(df.zscore<-3) | (df.zscore>3)]

# %% [markdown]
# <h3 style='color:purple'>Remove the outliers and produce new dataframe</h3>

# %%
df_no_outliers = df[(df.zscore>-3) & (df.zscore<3)]
df_no_outliers.head()

# %%
df_no_outliers.shape

# %%
df.shape

# %% [markdown]
# Above shows original dataframe data 10000 data points. Out of that we removed 7 outliers (i.e. 10000-9993) 

# %% [markdown]
# <h3 style='color:purple'>Exercise</h3>

# %% [markdown]
# You are given bhp.csv which contains property prices in the city of banglore, India. You need to examine price_per_sqft column and do following,
# 
# (1) Remove outliers using percentile technique first. Use [0.001, 0.999] for lower and upper bound percentiles
# 
# (2) After removing outliers in step 1, you get a new dataframe. 
# 
# (3) On step(2) dataframe, use 4 standard deviation to remove outliers
# 
# (4) Plot histogram for new dataframe that is generated after step (3). Also plot bell curve on same histogram
# 
# (5) On step(2) dataframe, use zscore of 4 to remove outliers. This is quite similar to step (3) and you will get exact same result

# %% [markdown]
# [Solution](https://github.com/codebasics/py/tree/master/ML/FeatureEngineering/2_outliers_z_score/Exercise/2_outliers_z_score_exercise.ipynb)


