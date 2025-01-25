 # [Script Title]: Programmatic Exploration of Ensemble Techniques  
 # Purpose: To demonstrate the effectiveness of different ensemble methods in machine learning  
 #  
 # Step 1: Import necessary libraries  
 # Step 2: Load a sample dataset for demonstration  
 # Step 3: Implement different ensemble techniques (Bagging, Boosting)  
 # Step 4: Evaluate and compare the performance of these techniques  
 #  
 # Introductory Example:  
 # We will use the Iris dataset to illustrate how Bagging and Boosting can improve classification accuracy.  
 
 import numpy as np  
 import pandas as pd  
 from sklearn.datasets import load_iris  
 from sklearn.model_selection import train_test_split  
 from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier  
 from sklearn.tree import DecisionTreeClassifier  
 from sklearn.metrics import accuracy_score  
 
 # Step 1: Load the Iris dataset  
 iris = load_iris()  
 X = iris.data  
 y = iris.target  
 
 # Step 2: Split the dataset into training and testing sets  
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
 
 # Step 3: Implement Bagging with Decision Trees  
 bagging_model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)  
 bagging_model.fit(X_train, y_train)  
 y_pred_bagging = bagging_model.predict(X_test)  
 bagging_accuracy = accuracy_score(y_test, y_pred_bagging)  
 print(f'Bagging Accuracy: {bagging_accuracy:.2f}')  
 
 # Step 4: Implement Boosting with Decision Trees  
 boosting_model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)  
 boosting_model.fit(X_train, y_train)  
 y_pred_boosting = boosting_model.predict(X_test)  
 boosting_accuracy = accuracy_score(y_test, y_pred_boosting)  
 print(f'Boosting Accuracy: {boosting_accuracy:.2f}')  
 
 # Conclusion: The script demonstrates how ensemble techniques can enhance model performance.  
 # Further exploration could include comparing more ensemble methods or tuning hyperparameters for better accuracy.