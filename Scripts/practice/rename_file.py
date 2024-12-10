import os

if os.path.exists('script.txt'):
    os.rename('script.txt', 'test.txt')
else:
    print("The file does not exist.")