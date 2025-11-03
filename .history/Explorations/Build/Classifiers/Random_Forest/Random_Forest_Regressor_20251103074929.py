"""
Exploration: Random Forest Regressor on California Housing Dataset
------------------------------------------------------------------

Objective:
1. Use sklearn’s California Housing dataset to predict median house value.
2. Train a Random Forest Regressor and evaluate performance using the R-squared metric [[R squared]].
3. Fine-tune the model by varying the number of trees (n_estimators) and observe changes in R².
"""

#---------------------------------------------
# Imports
#---------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

#---------------------------------------------
# Load Dataset
#---------------------------------------------
# California Housing: Predict median house value based on demographic and geographic features
data = fetch_california_housing()
df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['target'] = data['target']

print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"Feature names: {list(data['feature_names'])}")

#---------------------------------------------
# Define Features and Target
#---------------------------------------------
X = df.drop('target', axis=1)
y = df['target']

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#---------------------------------------------
# Model 1: Baseline Random Forest Regressor
#---------------------------------------------
print("\n=== Model 1: Baseline Random Forest Regressor (100 Trees) ===")

model_1 = RandomForestRegressor(n_estimators=100, random_state=42)
model_1.fit(X_train, y_train)

# Predict and evaluate
y_pred = model_1.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² Score: {r2:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

#---------------------------------------------
# Cross-Validation Evaluation
#---------------------------------------------
cv_scores = cross_val_score(model_1, X, y, cv=5, scoring='r2')
print(f"Mean Cross-Validation R²: {cv_scores.mean():.4f}")

#---------------------------------------------
# Fine-Tuning: Number of Trees (n_estimators)
#---------------------------------------------
print("\n=== Fine-Tuning Number of Trees (n_estimators) ===")

n_estimators_range = range(10, 210, 10)
r2_scores = []
cv_means = []

for n in n_estimators_range:
    model = RandomForestRegressor(n_estimators=n, random_state=42)
    model.fit(X_train, y_train)
    
    # R² on test data
    y_pred = model.predict(X_test)
    r2_scores.append(r2_score(y_test, y_pred))
    
    # Cross-validation mean R²
    cv_mean = cross_val_score(model, X, y, cv=5, scoring='r2').mean()
    cv_means.append(cv_mean)

# Identify best performing model
best_r2 = max(cv_means)
best_n_estimators = n_estimators_range[cv_means.index(best_r2)]
print(f"Best CV R²: {best_r2:.4f} with n_estimators={best_n_estimators}")

#---------------------------------------------
# Plot: R² vs Number of Trees
#---------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, r2_scores, marker='o', label='Test R²')
plt.plot(n_estimators_range, cv_means, marker='x', linestyle='--', label='CV Mean R²')
plt.title("Model Performance vs Number of Trees (n_estimators)")
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("R² Score")
plt.legend()
plt.grid()
plt.show()

#---------------------------------------------
# Feature Importance Analysis
#---------------------------------------------
print("\n=== Feature Importance ===")
feature_importances = pd.Series(model_1.feature_importances_, index=X.columns).sort_values(ascending=False)
print(feature_importances)

# Plot feature importance
feature_importances.plot(kind='bar', figsize=(10, 5), title='Feature Importance (Random Forest Regressor)')
plt.ylabel('Importance Score')
plt.show()

"""
Summary Notes:
--------------
- The Random Forest Regressor achieves R² ≈ 0.80 on the test set with 100 trees.
- Increasing n_estimators improves stability up to a point, after which gains are marginal.
- Cross-validation helps verify the model’s generalization.
- Feature importance highlights the most influential predictors in estimating house prices.
"""
