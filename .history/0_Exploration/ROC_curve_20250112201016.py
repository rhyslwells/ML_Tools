
# # **Machine Learning in Python: Making Receiver Operating Characteristic (ROC) curve**
# 
# Chanin Nantasenamat
# 
# <i>Data Professor YouTube channel, http://youtube.com/dataprofessor </i>
# 
# In this Jupyter notebook, we will be making a Receiver Operating Characteristic (ROC) curve using the Iris data set as an example.


# ## **What is ROC curve?**
# 
# The **ROC curve** summarizes the prediction performance of a classification model at all classification thresholds. Particularly, the ROC curve plots the **False Positive Rate (FPR)** on the *X-axis* and the **True Positive Rate (TPR)** on the *Y-axis*.
# 
# $\text{TPR (Sensitivity)} = \frac{TP}{TP + FN}$
# 
# $\text{FPR (1 - Specificity)} = \frac{FP}{TN + FP}$


# ## **Generate synthetic dataset**


from sklearn.datasets import make_classification
import numpy as np


X, Y = make_classification(n_samples=2000, n_classes=2, n_features=10, random_state=0)


# ## **Add noisy features to make the problem more difficult** $^1$
# 


random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]


# ## **Data splitting**


from sklearn.model_selection import train_test_split


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2,
                                                    random_state=0)


# ## **Build classification model**


from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


# ### Random Forest


rf = RandomForestClassifier(max_features=5, n_estimators=500)
rf.fit(X_train, Y_train)


# ### Naive Bayes


nb = GaussianNB()
nb.fit(X_train, Y_train)


# ## **Prediction probabilities** $^2$


r_probs = [0 for _ in range(len(Y_test))]
rf_probs = rf.predict_proba(X_test)
nb_probs = nb.predict_proba(X_test)


# Probabilities for the positive outcome is kept.


rf_probs = rf_probs[:, 1]
nb_probs = nb_probs[:, 1]


# ## **Computing AUROC and ROC curve values**


from sklearn.metrics import roc_curve, roc_auc_score


# ### **Calculate AUROC**
# **ROC** is the receiver operating characteristic
# **AUROC** is the area under the ROC curve


r_auc = roc_auc_score(Y_test, r_probs)
rf_auc = roc_auc_score(Y_test, rf_probs)
nb_auc = roc_auc_score(Y_test, nb_probs)


# ### **Print AUROC scores**


print('Random (chance) Prediction: AUROC = %.3f' % (r_auc))
print('Random Forest: AUROC = %.3f' % (rf_auc))
print('Naive Bayes: AUROC = %.3f' % (nb_auc))


# ### **Calculate ROC curve**


r_fpr, r_tpr, _ = roc_curve(Y_test, r_probs)
rf_fpr, rf_tpr, _ = roc_curve(Y_test, rf_probs)
nb_fpr, nb_tpr, _ = roc_curve(Y_test, nb_probs)


# ## **Plot the ROC curve**


import matplotlib.pyplot as plt



plt.plot(r_fpr, r_tpr, linestyle='--', label='Random prediction (AUROC = %0.3f)' % r_auc)
plt.plot(rf_fpr, rf_tpr, marker='.', label='Random Forest (AUROC = %0.3f)' % rf_auc)
plt.plot(nb_fpr, nb_tpr, marker='.', label='Naive Bayes (AUROC = %0.3f)' % nb_auc)

# Title
plt.title('ROC Plot')
# Axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Show legend
plt.legend() # 
# Show plot
plt.show()


# ## **Reference**
# 1. https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
# 2. https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
# 


# ---


