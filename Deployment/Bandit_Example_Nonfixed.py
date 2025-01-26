import subprocess
import os
import pickle

def dangerous_subprocess(user_input):
    """
    Executes a command using subprocess.
    
    Issues:
    - Command Injection: If `user_input` contains malicious shell commands, 
      it could execute unintended operations.
    - Example: 
    2+2
    "Hello".upper()
    """
    subprocess.call(f"cmd /c echo {user_input}", shell=True)

def hardcoded_password():
    """
    Stores a hardcoded password.
    
    Issues:
    - Hardcoding sensitive information is a security risk, as attackers can 
      extract it from the source code or memory dumps.
    - Example: Storing a database password like `password = "SuperSecret123"` 
      allows anyone with access to the code to compromise the database.
    """
    password = "SuperSecret123"  # Example of hardcoded sensitive information
    print(password)

def unsafe_eval(user_input):
    """
    Evaluates arbitrary Python code.

    Issues:
    - Code Execution: If `user_input` is malicious, it could execute harmful code.
    - Example: If `user_input` is `__import__('os').system('dir')`, it will list directory contents.
    """
    try:
        result = eval(user_input)  # Evaluate the input
        print(f"Result of eval: {result}")  # Print the result
    except Exception as e:
        print(f"Error during eval: {e}")  # Catch and print any errors
        
def insecure_deserialization(data):
    """
    Deserializes data using pickle.
    
    Issues:
    - Arbitrary Code Execution: Pickle allows execution of arbitrary code during deserialization.
    - Example: If `data` is crafted as `b"cos\nsystem\n(S'echo hello'\ntR."`, it will execute `os.system("echo hello")`.
    """
    return pickle.loads(data)  # If data is malicious, it can execute arbitrary code.

def main():
    """
    Main function to explore potential security issues interactively.
    
    Provides a menu to test each issue separately with simple examples.
    """
    print("\nChoose an issue to explore:")
    print("1. Dangerous Subprocess (Command Injection)")
    print("2. Hardcoded Password")
    print("3. Unsafe Eval (Code Execution)")
    print("4. Insecure Deserialization (Arbitrary Code Execution)")
    print("5. Exit")

    choice = input("Enter your choice (1-5): ").strip()
    
    if choice == "1":
        user_input = input("Enter something to echo: ")
        dangerous_subprocess(user_input)  # Simple Command Injection Example

    elif choice == "2":
        print("Hardcoded password example:")
        hardcoded_password()  # Just prints a comment about the hardcoded password

    elif choice == "3":
        user_input = input("Enter Python code to evaluate (e.g., '2 + 2'): ")
        unsafe_eval(user_input)  # Test safe or unsafe Python code

    elif choice == "4":
        print("Simulating insecure deserialization with safe data...")
        data = b'\x80\x03}q\x00(X\x03\x00\x00\x00keyq\x01X\x05\x00\x00\x00valueq\x02u.'
        result = insecure_deserialization(data)  # Deserialize a benign object
        print(f"Deserialized data: {result}")

    elif choice == "5":
        print("Exiting.")
    else:
        print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()