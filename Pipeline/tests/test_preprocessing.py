import pytest
import pandas as pd
from preprocessing import Preprocessor

@pytest.fixture
def sample_data():
    """Creates a sample dataframe with missing values."""
    return pd.DataFrame({"feature1": [1, None, 3], "feature2": [4, 5, None]})

def test_handle_missing_values(sample_data):
    preprocessor = Preprocessor()
    df_cleaned = preprocessor.handle_missing_values(sample_data)
    assert not df_cleaned.isnull().values.any()

def test_scale_features():
    preprocessor = Preprocessor()
    X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    X_test = pd.DataFrame({"feature1": [2, 3], "feature2": [5, 6]})
    
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
