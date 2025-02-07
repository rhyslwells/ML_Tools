import pytest
import pandas as pd
from dataset import Dataset

@pytest.fixture
def sample_data(tmp_path):
    """Creates a temporary dataset for testing."""
    file = tmp_path / "test_data.csv"
    df = pd.DataFrame({"feature1": [1, 2, 3, 4], "fraud": [0, 1, 0, 1]})
    df.to_csv(file, index=False)
    return str(file)

def test_load_data(sample_data):
    dataset = Dataset(sample_data)
    df = dataset.load_data()
    assert not df.empty
    assert "fraud" in df.columns

def test_train_test_split(sample_data):
    dataset = Dataset(sample_data)
    dataset.load_data()
    X_train, X_test, y_train, y_test = dataset.split("fraud")
    assert len(X_train) > 0
    assert len(y_train) > 0
