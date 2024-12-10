import os
import shutil

def organize_files_by_type(directory):
    """Organizes files in the given directory into subdirectories based on their file types."""
    try:
        # File type categories
        file_types = {
            'Images': ['.jpg', '.jpeg', '.png', '.gif'],
            'Documents': ['.pdf', '.docx', '.txt', '.xlsx'],
            'Videos': ['.mp4', '.avi', '.mov'],
            'Audio': ['.mp3', '.wav'],
            'Archives': ['.zip', '.rar', '.tar.gz']
        }
        
        # Create subdirectories and move files
        for filename in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, filename)):
                file_ext = os.path.splitext(filename)[1].lower()
                
                # Find the matching category
                for category, extensions in file_types.items():
                    if file_ext in extensions:
                        category_path = os.path.join(directory, category)
                        
                        # Create the subdirectory if it doesn't exist
                        if not os.path.exists(category_path):
                            os.makedirs(category_path)
                        
                        # Move the file to the subdirectory
                        shutil.move(
                            os.path.join(directory, filename),
                            os.path.join(category_path, filename)
                        )
                        print(f"Moved '{filename}' to '{category}/'")
                        break
        print("Organization completed.")
    except Exception as e:
        print(f"An error occurred: {e}")

# User input for directory
directory = input("Enter the path to the directory: ")

# Run the organization function
organize_files_by_type(directory)
