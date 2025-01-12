# **Machine Learning in Python: Receiver Operating Characteristic (ROC) Curve**

# Summary: This script demonstrates how to create and interpret a Receiver Operating Characteristic (ROC) curve, 
# showcasing the performance of classification models using the Iris dataset.

# Import necessary libraries
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

# ## **Key Concepts**
# - **ROC Curve**: A plot of the False Positive Rate (FPR) vs. True Positive Rate (TPR) for a classifier across thresholds.
# - **AUROC**: Area under the ROC curve, summarizing the classifier's performance (ranges from 0.5 for random guessing to 1 for perfect classification).
# - **TPR (Sensitivity)**: $TPR = \frac{TP}{TP + FN}$
# - **FPR (1 - Specificity)**: $FPR = \frac{FP}{TN + FP}$

# ## **1. Generate Synthetic Dataset**
X, Y = make_classification(n_samples=2000, n_classes=2, n_features=10, random_state=0)

# ## **2. Add Noisy Features**
# Adding noise to increase problem complexity
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# ## **3. Split the Data**
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# ## **4. Build Classification Models**
# ### Random Forest
rf = RandomForestClassifier(max_features=5, n_estimators=500)
rf.fit(X_train, Y_train)

# ### Naive Bayes
nb = GaussianNB()
nb.fit(X_train, Y_train)

# ## **5. Generate Prediction Probabilities**
# - Random predictions (chance baseline)
r_probs = [0 for _ in range(len(Y_test))]

# - Prediction probabilities for classifiers
rf_probs = rf.predict_proba(X_test)[:, 1]  # Random Forest
nb_probs = nb.predict_proba(X_test)[:, 1]  # Naive Bayes

# ## **6. Calculate AUROC and ROC Curve Values**
# ### AUROC Scores
r_auc = roc_auc_score(Y_test, r_probs)
rf_auc = roc_auc_score(Y_test, rf_probs)
nb_auc = roc_auc_score(Y_test, nb_probs)

# Print AUROC scores
print('Random (Chance) Prediction: AUROC = %.3f' % r_auc)
print('Random Forest: AUROC = %.3f' % rf_auc)
print('Naive Bayes: AUROC = %.3f' % nb_auc)

# ### ROC Curve Coordinates
r_fpr, r_tpr, _ = roc_curve(Y_test, r_probs)
rf_fpr, rf_tpr, _ = roc_curve(Y_test, rf_probs)
nb_fpr, nb_tpr, _ = roc_curve(Y_test, nb_probs)

# ## **7. Visualize the ROC Curve**
plt.figure(figsize=(8, 6))
# Random prediction
plt.plot(r_fpr, r_tpr, linestyle='--', label='Random prediction (AUROC = %0.3f)' % r_auc)
# Random Forest
plt.plot(rf_fpr, rf_tpr, marker='.', label='Random Forest (AUROC = %0.3f)' % rf_auc)
# Naive Bayes
plt.plot(nb_fpr, nb_tpr, marker='.', label='Naive Bayes (AUROC = %0.3f)' % nb_auc)

# Plot details
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend() 
plt.show()

# ## **Key Takeaways**
# - The ROC curve helps evaluate the trade-off between TPR and FPR across thresholds.
# - AUROC provides a single metric to compare classifiers, with higher values indicating better performance.

# ## **References**
# 1. Scikit-learn documentation: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
# 2. Machine Learning Mastery: https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
