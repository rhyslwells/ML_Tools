import pytest
from evaluation import ModelEvaluator

def test_evaluate():
    evaluator = ModelEvaluator()
    y_test = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]
    
    accuracy = evaluator.evaluate("logistic_regression", y_test, y_pred)
    assert 0 <= accuracy <= 1

def test_best_model():
    evaluator = ModelEvaluator()
    evaluator.results = {"logistic_regression": 0.9, "random_forest": 0.95}
    
    best_model, best_acc = evaluator.best_model()
    assert best_model == "random_forest"
    assert best_acc == 0.95
