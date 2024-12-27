import cv2
import os
import numpy as np

# Dataset path
dataset_path = "dataset"
class_names = ["sneakers", "sandals", "formal_shoes", "simple_shoes", "slippers"]

# Folder to save preprocessed images
preprocessed_path = "preprocessed_dataset"
os.makedirs(preprocessed_path, exist_ok=True)

# Loop through each class folder
for label, class_name in enumerate(class_names):
    class_folder = os.path.join(dataset_path, class_name)
    preprocessed_class_folder = os.path.join(preprocessed_path, class_name)
    os.makedirs(preprocessed_class_folder, exist_ok=True)  # Create folder for each class in preprocessed dataset
    
    for image_name in os.listdir(class_folder):
        image_path = os.path.join(class_folder, image_name)
        
        # Load image in RGB format (not grayscale)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load image in color (RGB)
        
        # Resize image to 128x128 or any size you prefer
        img = cv2.resize(img, (256, 256))  # Increase the image dimensions
        
        # Save preprocessed image
        preprocessed_image_path = os.path.join(preprocessed_class_folder, image_name)
        cv2.imwrite(preprocessed_image_path, img)

print("Preprocessed images saved to:", preprocessed_path)
