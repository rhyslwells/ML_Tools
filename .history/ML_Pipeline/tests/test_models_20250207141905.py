import pytest
from models import ModelFactory

def test_create_models():
    """Ensure models are initialized correctly."""
    model_logistic = ModelFactory("logistic_regression")
    model_rf = ModelFactory("random_forest")

    assert model_logistic.model is not None
    assert model_rf.model is not None

def test_model_training():
    """Ensure models train without errors."""
    model = ModelFactory("logistic_regression")
    X_train = [[1, 2], [3, 4], [5, 6]]
    y_train = [0, 1, 0]
    
    model.train(X_train, y_train)
    assert model.model is not None
