import os

# Define the dataset directory and new folder names (classes)
dataset_path = "dataset"
class_names = ["sneakers", "slippers",  "formal_shoes", "simple_shoes","sandals"]

# Get the list of folders in the dataset directory
folders = sorted(os.listdir(dataset_path))

if len(folders) != len(class_names):
    print("Error: Number of folders does not match the number of class names!")
    exit()

# Rename folders and their contents
for i, folder in enumerate(folders):
    old_folder_path = os.path.join(dataset_path, folder)
    new_folder_path = os.path.join(dataset_path, class_names[i])
    
    # Rename the folder
    os.rename(old_folder_path, new_folder_path)
    print(f"Renamed folder '{folder}' to '{class_names[i]}'")
    
    # Rename the images inside the folder
    images = os.listdir(new_folder_path)
    for j, image in enumerate(images, start=1):
        old_image_path = os.path.join(new_folder_path, image)
        new_image_name = f"{class_names[i]}_{j}.jpg"
        new_image_path = os.path.join(new_folder_path, new_image_name)
        
        os.rename(old_image_path, new_image_path)
        print(f"Renamed '{image}' to '{new_image_name}'")

print("All folders and images have been renamed successfully!")
