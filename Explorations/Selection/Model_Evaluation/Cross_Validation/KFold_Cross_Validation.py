# %% [markdown]
# <h1 style='color:blue;' align='center'>KFold Cross Validation Python Tutorial</h2>

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
digits = load_digits()

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.3)

# %% [markdown]
# **Logistic Regression**

# %%
lr = LogisticRegression(solver='liblinear',multi_class='ovr')
lr.fit(X_train, y_train)
lr.score(X_test, y_test)

# %% [markdown]
# **SVM**

# %%
svm = SVC(gamma='auto')
svm.fit(X_train, y_train)
svm.score(X_test, y_test)

# %% [markdown]
# **Random Forest**

# %%
rf = RandomForestClassifier(n_estimators=40)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)

# %% [markdown]
# <h2 style='color:purple'>KFold cross validation</h2>

# %% [markdown]
# **Basic example**

# %%
from sklearn.model_selection import KFold
kf = KFold(n_splits=3)
kf

# %%
for train_index, test_index in kf.split([1,2,3,4,5,6,7,8,9]):
    print(train_index, test_index)

# %% [markdown]
# **Use KFold for our digits example**

# %%
def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

# %%
from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold(n_splits=3)

scores_logistic = []
scores_svm = []
scores_rf = []

for train_index, test_index in folds.split(digits.data,digits.target):
    X_train, X_test, y_train, y_test = digits.data[train_index], digits.data[test_index], \
                                       digits.target[train_index], digits.target[test_index]
    scores_logistic.append(get_score(LogisticRegression(solver='liblinear',multi_class='ovr'), X_train, X_test, y_train, y_test))  
    scores_svm.append(get_score(SVC(gamma='auto'), X_train, X_test, y_train, y_test))
    scores_rf.append(get_score(RandomForestClassifier(n_estimators=40), X_train, X_test, y_train, y_test))

# %%
scores_logistic

# %%
scores_svm

# %%
scores_rf

# %% [markdown]
# <h2 style='color:purple'>cross_val_score function</h2>

# %%
from sklearn.model_selection import cross_val_score

# %% [markdown]
# **Logistic regression model performance using cross_val_score**

# %%
cross_val_score(LogisticRegression(solver='liblinear',multi_class='ovr'), digits.data, digits.target,cv=3)

# %% [markdown]
# **svm model performance using cross_val_score**

# %%
cross_val_score(SVC(gamma='auto'), digits.data, digits.target,cv=3)

# %% [markdown]
# **random forest performance using cross_val_score**

# %%
cross_val_score(RandomForestClassifier(n_estimators=40),digits.data, digits.target,cv=3)

# %% [markdown]
# cross_val_score uses stratifield kfold by default

# %% [markdown]
# <h2 style='color:purple'>Parameter tunning using k fold cross validation</h2>

# %%
scores1 = cross_val_score(RandomForestClassifier(n_estimators=5),digits.data, digits.target, cv=10)
np.average(scores1)

# %%
scores2 = cross_val_score(RandomForestClassifier(n_estimators=20),digits.data, digits.target, cv=10)
np.average(scores2)

# %%
scores3 = cross_val_score(RandomForestClassifier(n_estimators=30),digits.data, digits.target, cv=10)
np.average(scores3)

# %%
scores4 = cross_val_score(RandomForestClassifier(n_estimators=40),digits.data, digits.target, cv=10)
np.average(scores4)

# %% [markdown]
# Here we used cross_val_score to
# fine tune our random forest classifier and figured that having around 40 trees in random forest gives best result. 

# %% [markdown]
# <h2 style='color:purple'>Exercise</h2>

# %% [markdown]
# Use iris flower dataset from sklearn library and use cross_val_score against following
# models to measure the performance of each. In the end figure out the model with best performance,
# 1. Logistic Regression
# 2. SVM
# 3. Decision Tree
# 4. Random Forest


# %%
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# %%
iris = load_iris()

# %% [markdown]
# **Logistic Regression**

# %%
l_scores = cross_val_score(LogisticRegression(), iris.data, iris.target)
l_scores

# %%
np.average(l_scores)

# %% [markdown]
# **Decision Tree**

# %%
d_scores = cross_val_score(DecisionTreeClassifier(), iris.data, iris.target)
d_scores

# %%
np.average(d_scores)

# %% [markdown]
# **Support Vector Machine (SVM)**

# %%
s_scores = cross_val_score(SVC(), iris.data, iris.target)
s_scores

# %%
np.average(s_scores)

# %% [markdown]
# **Random Forest**

# %%
r_scores = cross_val_score(RandomForestClassifier(n_estimators=40), iris.data, iris.target)
r_scores

# %%
np.average(r_scores)

# %% [markdown]
# **Best score so far is from SVM: 0.97344771241830064**




```python
from sklearn.model_selection import cross_val_score
cross_val_score(model, X_train, y_train, cv=5)
```

Implement k-fold cross-validation on a dataset and use it to train and evaluate a machine learning model:

```python
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# Create a dataframe with sample data
data = {'X1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'X2': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'X3': [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        'y': [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]}
df = pd.DataFrame(data)

# Split the data into features and target
X = df.drop('y', axis=1)
y = df['y']

# Create a linear regression model
model = LinearRegression()

# Perform k-fold cross-validation
scores = cross_val_score(model, X, y, cv=5)

# Print the scores
print(scores)
print("Mean Score:", scores.mean())
```

If the mean score is close to 1 (e.g., 0.96), the model appears to perform consistently well across most folds, with an average accuracy of approximately 96%.

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

model_2 = RandomForestClassifier(n_estimators=10)

# Perform cross-validation
cv_scores = cross_val_score(model_2, X, y, cv=5)

# Print cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())
```
