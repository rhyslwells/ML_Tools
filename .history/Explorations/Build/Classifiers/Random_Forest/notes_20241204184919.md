# Sklearn Datasets

Sklearn Datasets are a collection of datasets used 
for testing machine learning algorithms.

with each dataset, we have keys:
- data: np array 
- target
- feature_names
- DESCR
- images
- column names: texxtual ordered column names
- ect

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

it is ordered nicely.

Why is is useful. What are some interest examples.

Time series, and updating model using new information. When to train.

# Robustness and testing? 

What is the impact on the random state on these models? plot 


https://scikit-learn.org/dev/glossary.html#term-random_state

An integer
Use a new random number generator seeded by the given integer. Using an int will produce the same results across different calls. However, it may be worthwhile checking that your results are stable across a number of different distinct random seeds. Popular integer random seeds are 0 and 42. Integer values must be in the range [0, 2**32 - 1].

https://scikit-learn.org/dev/common_pitfalls.html#randomness

10.3.2.1. Estimators
Different `random_state` types lead to different cross-validation procedures

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
import numpy as np

X, y = make_classification(random_state=0)

rf_123 = RandomForestClassifier(random_state=123)
cross_val_score(rf_123, X, y)
array([0.85, 0.95, 0.95, 0.9 , 0.9 ])

rf_inst = RandomForestClassifier(random_state=np.random.RandomState(0))
cross_val_score(rf_inst, X, y)
array([0.9 , 0.95, 0.95, 0.9 , 0.9 ])
We see that the cross-validated scores of rf_123 and rf_inst are different, as should be expected since we didn’t pass the same random_state parameter. However, the difference between these scores is more subtle than it looks, and the cross-validation procedures that were performed by cross_val_score significantly differ in each case:

Since rf_123 was passed an integer, every call to fit uses the same RNG: this means that all random characteristics of the random forest estimator will be the same for each of the 5 folds of the CV procedure. In particular, the (randomly chosen) subset of features of the estimator will be the same across all folds.

Since rf_inst was passed a RandomState instance, each call to fit starts from a different RNG. As a result, the random subset of features will be different for each folds.

While having a constant estimator RNG across folds isn’t inherently wrong, 

--- we usually want CV results that are robust w.r.t. the estimator’s randomness.--- [[Cross validation]] note.
from sklearn.model_selection import cross_val_score

 As a result, passing an instance instead of an integer may be preferable, since it will allow the estimator RNG to vary for each fold.

 cross validation: Here, cross_val_score will use a non-randomized CV splitter (as is the default), so both estimators will be evaluated on the same splits.


Evaluating Estimator Sensitivity:

By plotting how accuracy varies with the number of estimators for different seeds and folds, you can detect patterns or irregularities, such as:
A minimum number of trees required to stabilize accuracy.
A plateau beyond which adding more estimators doesn’t improve performance.
'''
