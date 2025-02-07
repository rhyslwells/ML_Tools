import unittest
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

class TestMathOperations(unittest.TestCase):
    """
    Test case for math operations.
    """

    def test_add(self):
        # Test the add function
        self.assertEqual(add(3, 4), 7)  # This will fail
        self.assertEqual(add(-1, 1), 0)  # This will fail
        self.assertEqual(add(0, 0), 0)  # This will fail

    def test_subtract(self):
        # Test the subtract function
        self.assertEqual(subtract(10, 5), 5)
        self.assertEqual(subtract(-1, -1), 0)
        self.assertEqual(subtract(0, 5), -5)

    def test_multiply(self):
        # Test the multiply function
        self.assertEqual(multiply(3, 4), 12)
        self.assertEqual(multiply(-1, 1), -1)
        self.assertEqual(multiply(0, 5), 0)

    def test_divide(self):
        # Test the divide function
        self.assertEqual(divide(10, 2), 5)
        self.assertEqual(divide(-10, 2), -5)
        self.assertEqual(divide(0, 1), 0)

        # Test division by zero
        with self.assertRaises(ValueError):
            divide(10, 0)

    @patch('__main__.add')
    def test_mock_add(self, mock_add):
        # Mock the add function to return a fixed value
        mock_add.return_value = 10
        self.assertEqual(add(3, 4), 10)
        mock_add.assert_called_once_with(3, 4)

if __name__ == '__main__':
    # Run the tests
    unittest.main()