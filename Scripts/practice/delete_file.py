import os

if os.path.exists('script.txt'):
    os.remove('script.txt')
else:
    print("The file does not exist.")