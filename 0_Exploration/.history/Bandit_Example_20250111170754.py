import subprocess
import os
import pickle

def dangerous_subprocess(user_input):
    # Runs a subprocess using the Windows command prompt
    subprocess.call(f"cmd /c echo {user_input}", shell=True)

def hardcoded_password():
    # Hardcoded password issue
    password = "SuperSecret123"

def unsafe_eval(user_input):
    # This demonstrates the use of eval but expects valid Python code as input
    try:
        result = eval(user_input)
        print(f"Result of eval: {result}")
    except Exception as e:
        print(f"Error in eval: {e}")

def insecure_deserialization(data):
    # Insecure deserialization with pickle
    return pickle.loads(data)

def main():
    user_input = input("Enter something to be evaluated in python: ")
    
    # Call functions that demonstrate potential security issues
    dangerous_subprocess(user_input)
    hardcoded_password()
    unsafe_eval(user_input)
    
    # Simulate insecure deserialization
    data = b'\x80\x03}q\x00(X\x03\x00\x00\x00keyq\x01X\x05\x00\x00\x00valueq\x02u.'
    insecure_deserialization(data)

if __name__ == "__main__":
    main()

# "Hello".upper()
# 2+2