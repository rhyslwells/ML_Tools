from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from config import Config

class ModelFactory:
    def __init__(self, model_type):
        """Initializes a machine learning model based on the selected type."""
        if model_type == "logistic_regression":
            self.model = LogisticRegression(**Config.MODELS[model_type])
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(**Config.MODELS[model_type])
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train(self, X_train, y_train):
        """Trains the selected model."""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions on new data."""
        return self.model.predict(X_test)
