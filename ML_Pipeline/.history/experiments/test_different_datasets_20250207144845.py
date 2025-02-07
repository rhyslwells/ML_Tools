# experiments/test_different_datasets.py

import pandas as pd
import sys
import os

# Ensure we can import from the scripts folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

import dataset  # Refers to scripts/dataset.py
import preprocessing  # Refers to scripts/preprocessing.py
import models  # Refers to scripts/models.py
import evaluation  # Refers to scripts/evaluation.py
import utils  # Refers to scripts/utils.py

# Path to the data folder
data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')

datasets = [
    "fraud_data.csv",       # Default dataset
    "customer_transactions.csv",  # Another dataset for testing
    "synthetic_fraud.csv"   # Another dataset for testing
]

for dataset_name in datasets:
    dataset_path = os.path.join(data_folder, dataset_name)
    
    utils.log_message(f"\n📌 Testing dataset: {dataset_path}")

    # Load data
    data = dataset.Dataset(dataset_path)
    df = data.load_data()

    if "fraud" not in df.columns:
        utils.log_message(f"Skipping {dataset_name}, missing 'fraud' column.")
        continue

    X_train, X_test, y_train, y_test = data.split(target_column="fraud")

    # Preprocess data
    preprocessor = preprocessing.Preprocessor()
    X_train = preprocessor.handle_missing_values(X_train)
    X_test = preprocessor.handle_missing_values(X_test)
    X_train, X_test = preprocessor.scale_features(X_train, X_test)

    # Train models
    models_list = ["logistic_regression", "random_forest"]
    evaluator = evaluation.ModelEvaluator()

    for model_name in models_list:
        utils.log_message(f"Training {model_name} on {dataset_name}...")
        model = models.ModelFactory(model_name)
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = evaluator.evaluate(model_name, y_test, y_pred)
        utils.log_message(f"{model_name} Accuracy on {dataset_name}: {acc:.2f}")

    # Select best model
    best_model, best_acc = evaluator.best_model()
    utils.log_message(f"Best Model for {dataset_name}: {best_model} with Accuracy: {best_acc:.2f}")
