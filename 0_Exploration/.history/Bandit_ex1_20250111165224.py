import subprocess

user_input = input("Enter your name: ")
subprocess.call(["echo", user_input], shell=False)