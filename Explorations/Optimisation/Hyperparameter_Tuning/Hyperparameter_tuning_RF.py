
```python
# **4. Hyperparameter Tuning**

Now we will be performing the tuning of hyperparameters of Random forest model. The hyperparameters that we will tune includes **max_features** and the **n_estimators**.

Note: Some codes modified from [scikit-learn](http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html)

Firstly, we will import the necessary modules.

The **GridSearchCV()** function from scikit-learn will be used to perform the hyperparameter tuning. Particularly, **GridSearchCV()** function can perform the typical functions of a classifier such as _**fit**_, _**score**_ and _**predict**_ as well as _**predict_proba**_, _**decision_function**_, _**transform**_ and _**inverse_transform**_.

Secondly, we define variables that are necessary input to the GridSearchCV() function.

In [0]:

from sklearn.model_selection import GridSearchCV
import numpy as np

max_features_range = np.arange(1,6,1)
n_estimators_range = np.arange(10,210,10)
param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)

rf = RandomForestClassifier()

grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
```


