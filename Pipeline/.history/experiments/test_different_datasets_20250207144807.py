import pandas as pd
from scripts.dataset import Dataset
from scripts.preprocessing import Preprocessor
from scripts.models import ModelFactory
from scripts.evaluation import ModelEvaluator
from scripts.utils import log_message

# List of datasets to test
datasets = [
    "fraud_data.csv",       # Default dataset
    "customer_transactions.csv",  # New dataset
    "synthetic_fraud.csv"   # Another dataset for testing
]

for dataset_path in datasets:
    log_message(f"\n📌 Testing dataset: {dataset_path}")

    # Load Data
    dataset = Dataset(dataset_path)
    df = dataset.load_data()
    
    if "fraud" not in df.columns:
        log_message(f"Skipping {dataset_path}, missing 'fraud' column.")
        continue  # Skip if the target variable is missing
    
    X_train, X_test, y_train, y_test = dataset.split(target_column="fraud")

    # Preprocess Data
    preprocessor = Preprocessor()
    X_train = preprocessor.handle_missing_values(X_train)
    X_test = preprocessor.handle_missing_values(X_test)
    X_train, X_test = preprocessor.scale_features(X_train, X_test)

    # Train Models
    models = ["logistic_regression", "random_forest"]
    evaluator = ModelEvaluator()

    for model_name in models:
        log_message(f"Training {model_name} on {dataset_path}...")
        model = ModelFactory(model_name)
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = evaluator.evaluate(model_name, y_test, y_pred)
        log_message(f"{model_name} Accuracy on {dataset_path}: {acc:.2f}")

    # Select Best Model
    best_model, best_acc = evaluator.best_model()
    log_message(f"Best Model for {dataset_path}: {best_model} with Accuracy: {best_acc:.2f}")
