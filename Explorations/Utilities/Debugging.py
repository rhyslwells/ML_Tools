import logging

# Configure logging to output messages to the console
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def divide_numbers(num1, num2):
    """
    Function to divide two numbers and demonstrate debugging techniques.
    """
    logging.debug(f"Attempting to divide {num1} by {num2}")
    
    try:
        # Intentional bug: division by zero
        result = num1 / num2
        logging.info(f"Division successful: {result}")
        return result
    except ZeroDivisionError as e:
        logging.error("Error: Division by zero is not allowed.")
        return None

def main():
    # Example of a bug: dividing by zero
    num1 = 10
    num2 = 0  # Change this to a non-zero number to fix the bug

    # Set a breakpoint here to inspect variables
    result = divide_numbers(num1, num2)
    
    # Log the result
    if result is not None:
        logging.info(f"The result of division is: {result}")
    else:
        logging.warning("The division could not be performed.")

if __name__ == "__main__":
    # Run the main function
    main()

    # Example of using a profiler or static analysis tool
    # Uncomment the following line to simulate a static analysis warning
    # unused_variable = 42

    # Example of a simple automated test
    assert divide_numbers(10, 2) == 5, "Test failed: 10 divided by 2 should be 5"
    logging.info("All tests passed.")