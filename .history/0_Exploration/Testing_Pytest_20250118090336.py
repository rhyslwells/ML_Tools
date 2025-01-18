import pytest
from unittest.mock import patch

def add(a, b):
    """
    Function to add two numbers.
    """
    return a + b + 1  # Intentional error: adding 1 to the result

def subtract(a, b):
    """
    Function to subtract two numbers.
    """
    return a - b

def multiply(a, b):
    """
    Function to multiply two numbers.
    """
    return a * b

def divide(a, b):
    """
    Function to divide two numbers.
    Raises a ValueError if division by zero is attempted.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def test_add():
    # Test the add function
    assert add(3, 4) == 7  # This will fail
    assert add(-1, 1) == 0  # This will fail
    assert add(0, 0) == 0  # This will fail

def test_subtract():
    # Test the subtract function
    assert subtract(10, 5) == 5
    assert subtract(-1, -1) == 0
    assert subtract(0, 5) == -5

def test_multiply():
    # Test the multiply function
    assert multiply(3, 4) == 12
    assert multiply(-1, 1) == -1
    assert multiply(0, 5) == 0

def test_divide():
    # Test the divide function
    assert divide(10, 2) == 5
    assert divide(-10, 2) == -5
    assert divide(0, 1) == 0

    # Test division by zero
    with pytest.raises(ValueError):
        divide(10, 0)

@patch('__main__.add')
def test_mock_add(mock_add):
    # Mock the add function to return a fixed value
    mock_add.return_value = 10
    assert add(3, 4) == 10
    mock_add.assert_called_once_with(3, 4)
