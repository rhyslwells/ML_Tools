from sklearn.metrics import accuracy_score

class ModelEvaluator:
    def __init__(self):
        self.results = {}

    def evaluate(self, model_name, y_test, y_pred):
        """Evaluates the model and stores accuracy."""
        accuracy = accuracy_score(y_test, y_pred)
        self.results[model_name] = accuracy
        return accuracy

    def best_model(self):
        """Returns the best performing model."""
        return max(self.results, key=self.results.get), max(self.results.values())
