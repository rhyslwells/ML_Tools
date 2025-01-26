import subprocess
import os
import ast
import json

def dangerous_subprocess(user_input):
    """
    Safely executes a command using subprocess.

    Why this is better:
    - Instead of using `shell=True`, which allows arbitrary command execution,
      we use `shell=False` with a list of arguments to ensure the command is
      executed as intended without invoking the shell.
    - Basic sanitization is performed on the user input to remove dangerous characters.
    """
    sanitized_input = user_input.replace('"', '').replace("'", "")  # Basic sanitization
    subprocess.call(["cmd", "/c", "echo", sanitized_input], shell=False)

def hardcoded_password():
    """
    Safely fetches a password from an environment variable.

    Why this is better:
    - Instead of hardcoding sensitive information like passwords in the source code,
      this approach retrieves it from an environment variable.
    - Using environment variables ensures that sensitive information is not exposed in the codebase
      or version control systems.
    """
    password = os.getenv("MY_APP_PASSWORD", "DefaultPassword123")  # Use environment variable
    print(f"Password: {password if password != 'DefaultPassword123' else 'Not Set'}")

def unsafe_eval(user_input):
    """
    Safely evaluates Python literals.

    Why this is better:
    - Instead of using `eval`, which can execute arbitrary code and pose a major security risk,
      `ast.literal_eval` is used. This method is limited to evaluating Python literals such as
      numbers, strings, tuples, lists, dictionaries, and sets.
    - This approach ensures that code execution is not possible while still allowing structured data parsing.
    """
    try:
        result = ast.literal_eval(user_input)  # Evaluate only literals
        print(f"Result of eval: {result}")
    except (ValueError, SyntaxError) as e:
        print(f"Error during eval: {e}")

def insecure_deserialization(data):
    """
    Safely deserializes data using JSON instead of pickle.

    Why this is better:
    - Pickle is inherently insecure because it can execute arbitrary code during deserialization,
      making it vulnerable to code injection attacks.
    - JSON is a safer alternative for data serialization and deserialization as it does not support
      code execution.
    - JSON is widely used, more human-readable, and less prone to misuse.
    """
    try:
        result = json.loads(data.decode("utf-8"))  # Deserialize JSON data
        return result
    except json.JSONDecodeError as e:
        print(f"Deserialization error: {e}")
        return None

def main():
    """
    Main function to explore potential security issues interactively with fixed implementations.

    Why this structure is better:
    - Each issue is isolated, allowing focused testing and demonstration.
    - The menu structure makes it easy to navigate and understand the context of each issue.
    """
    while True:
        print("\nChoose an issue to explore:")
        print("1. Dangerous Subprocess (Command Injection)")
        print("2. Hardcoded Password")
        print("3. Unsafe Eval (Code Execution)")
        print("4. Insecure Deserialization (Arbitrary Code Execution)")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == "1":
            user_input = input("Enter something to echo: ")
            dangerous_subprocess(user_input)  # Fixed Subprocess Example

        elif choice == "2":
            print("Secure hardcoded password example:")
            hardcoded_password()  # Secure Password Management

        elif choice == "3":
            user_input = input("Enter Python literal to evaluate (e.g., '[1, 2, 3]'): ")
            unsafe_eval(user_input)  # Safer Evaluation Example

        elif choice == "4":
            print("Simulating secure deserialization with JSON data...")
            data = b'{"key": "value"}'  # JSON data
            result = insecure_deserialization(data)  # Safe Deserialization
            print(f"Deserialized data: {result}")

        elif choice == "5":
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
