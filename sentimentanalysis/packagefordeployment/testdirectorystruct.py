import os   
print('manish shukla') 
print(os.getcwd())
with os.scandir(os.getcwd()) as entries:
    for entry in entries:
        print(entry.name)