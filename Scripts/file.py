import os

def rename_files(directory, prefix='', suffix=''):
    """Renames all files in the specified directory by adding a prefix and/or suffix."""
    try:
        # List all files in the given directory
        files = os.listdir(directory)
        
        for filename in files:
            # Skip directories, only rename files
            if os.path.isfile(os.path.join(directory, filename)):
                # Split the file into name and extension
                name, ext = os.path.splitext(filename)
                
                # Construct the new filename
                new_name = f"{prefix}{name}{suffix}{ext}"
                
                # Rename the file
                os.rename(
                    os.path.join(directory, filename),
                    os.path.join(directory, new_name)
                )
                print(f"Renamed '{filename}' to '{new_name}'")
        print("Renaming completed.")
    except Exception as e:
        print(f"An error occurred: {e}")

# User input for directory, prefix, and suffix
directory = input("Enter the path to the directory: ")
prefix = input("Enter the prefix (leave blank if none): ")
suffix = input("Enter the suffix (leave blank if none): ")

# Run the renaming function
rename_files(directory, prefix, suffix)
