from dataset import Dataset
from preprocessing import Preprocessor
from models import ModelFactory
from evaluation import ModelEvaluator
from utils import log_message

# Step 1: Load Data
dataset = Dataset("fraud_data.csv")
df = dataset.load_data()
X_train, X_test, y_train, y_test = dataset.split(target_column="fraud")

# Step 2: Preprocess Data
preprocessor = Preprocessor()
X_train = preprocessor.handle_missing_values(X_train)
X_test = preprocessor.handle_missing_values(X_test)
X_train, X_test = preprocessor.scale_features(X_train, X_test)

# Step 3: Train Models
models = ["logistic_regression", "random_forest"]
evaluator = ModelEvaluator()

for model_name in models:
    log_message(f"Training {model_name}...")
    model = ModelFactory(model_name)
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = evaluator.evaluate(model_name, y_test, y_pred)
    log_message(f"{model_name} Accuracy: {acc:.2f}")

# Step 4: Select Best Model
best_model, best_acc = evaluator.best_model()
log_message(f"\nBest Model: {best_model} with Accuracy: {best_acc:.2f}")
