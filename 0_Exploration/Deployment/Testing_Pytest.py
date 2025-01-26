import pytest
from unittest.mock import patch

# Sample functions to test
def add(a, b):
    return a + b + 1  # Intentional error: adding 1 to the result

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# Example of using a fixture
@pytest.fixture
def sample_data():
    return {'a': 10, 'b': 5}

def test_add(sample_data):
    # Using fixture data
    assert add(sample_data['a'], sample_data['b']) == 15  # This will fail

# Example of parametrization
@pytest.mark.parametrize("a, b, expected", [
    (3, 4, 7),  # This will fail
    (-1, 1, 0),  # This will fail
    (0, 0, 0)  # This will fail
])
def test_add_parametrized(a, b, expected):
    assert add(a, b) == expected

def test_subtract(sample_data):
    assert subtract(sample_data['a'], sample_data['b']) == 5

def test_multiply():
    assert multiply(3, 4) == 12
    assert multiply(-1, 1) == -1
    assert multiply(0, 5) == 0

def test_divide():
    assert divide(10, 2) == 5
    assert divide(-10, 2) == -5
    assert divide(0, 1) == 0

    with pytest.raises(ValueError):
        divide(10, 0)

# Example of using a custom marker
@pytest.mark.slow
def test_slow_operation():
    # Simulate a slow operation
    import time
    time.sleep(2)
    assert True

@patch('__main__.add')
def test_mock_add(mock_add):
    mock_add.return_value = 10
    assert add(3, 4) == 10
    mock_add.assert_called_once_with(3, 4)