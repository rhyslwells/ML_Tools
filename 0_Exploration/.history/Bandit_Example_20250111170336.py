import subprocess
import os
import pickle

def dangerous_subprocess(user_input):
    # Potential command injection vulnerability
    subprocess.call(f"echo {user_input}", shell=True)

def hardcoded_password():
    # Hardcoded password issue
    password = "SuperSecret123"

def unsafe_eval(user_input):
    # Use of eval can lead to code execution vulnerabilities
    eval(user_input)

def insecure_deserialization(data):
    # Insecure deserialization with pickle
    return pickle.loads(data)

def main():
    user_input = input("Enter something: ")
    
    # Call functions that demonstrate potential security issues
    dangerous_subprocess(user_input)
    hardcoded_password()
    unsafe_eval(user_input)
    
    # Simulate insecure deserialization
    data = b'\x80\x03}q\x00(X\x03\x00\x00\x00keyq\x01X\x05\x00\x00\x00valueq\x02u.'
    insecure_deserialization(data)

if __name__ == "__main__":
    main()