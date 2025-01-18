import unittest

def add(a, b):
    """
    Function to add two numbers.
    """
    return a + b

def subtract(a, b):
    """
    Function to subtract two numbers.
    """
    return a - b

class TestMathOperations(unittest.TestCase):
    """
    Test case for math operations.
    """

    def test_add(self):
        # Test the add function
        self.assertEqual(add(3, 4), 7)
        self.assertEqual(add(-1, 1), 0)
        self.assertEqual(add(0, 0), 0)

    def test_subtract(self):
        # Test the subtract function
        self.assertEqual(subtract(10, 5), 5)
        self.assertEqual(subtract(-1, -1), 0)
        self.assertEqual(subtract(0, 5), -5)

if __name__ == '__main__':
    # Run the tests
    unittest.main()