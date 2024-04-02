   
# 3. **Answering Specific Questions**:
#    - Address questions like identifying companies posting the same job title multiple times and reasons behind it. You've started doing this by grouping job titles by company ID.
#    - Analyze job classifications and their relationships with other factors such as views, applications, and median salaries.


#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('job_postings.csv')


# company analysis
# give a company score based on the number of job postings
# ['posting_effectiveness']






# ## Analysis with salary
# 

# Does the highest paying jobs attact the most view and or apps? (scatter pay to views and apps).

# What is the distrubition of applications/views against pay?

# is the pay_period (hrly) indictive of lower paying jobs?
## Boxplot for salary information
# plt.figure(figsize=(12, 8))
# sns.boxplot(x='pay_period', y='max_salary', data=df)
# plt.title('Boxplot of Max Salary by Pay Period')
# plt.show()
