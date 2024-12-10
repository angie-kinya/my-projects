import getpass

password = getpass.getpass("Enter the password to access this script: ")

if password != "your_password":
    print("Access denied.")
    exit()

print("Access granted. Running the script...")