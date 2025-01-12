import subprocess

user_input = input("Enter your name: ")
subprocess.call(f"echo {user_input}", shell=True)
