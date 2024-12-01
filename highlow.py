import cv2
import numpy as np
import os

def calculate_brightness_percentage(image_path):
    # Read the image (already in grayscale)
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Consider very dark pixels as fluid
    # Pixels below this threshold are considered "dark" (fluid)
    dark_threshold = 35
    
    # Create mask for dark pixels (fluid areas)
    dark_mask = gray < dark_threshold
    
    # Calculate percentage of dark pixels (fluid)
    total_pixels = gray.size
    dark_pixels = np.sum(dark_mask)
    percentage = (dark_pixels / total_pixels) * 100
    
    return percentage

def categorize_images(data_folder):
    for filename in os.listdir(data_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(data_folder, filename)
            
            # Calculate dark (fluid) percentage
            fluid_percentage = calculate_brightness_percentage(file_path)
            
            # Get file name without extension
            name, ext = os.path.splitext(filename)
            
            # Remove any existing category suffix if present
            for suffix in ['_high', '_normal', '_low']:
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
            
            # Determine category based on fluid percentage
            if fluid_percentage >= 60:
                new_name = f"{name}_high{ext}"  # High fluid
            elif fluid_percentage < 40:
                new_name = f"{name}_low{ext}"   # Low fluid
            else:
                new_name = f"{name}_normal{ext}" # Normal fluid
            
            # Rename the file
            new_path = os.path.join(data_folder, new_name)
            os.rename(file_path, new_path)
            print(f"Processed {filename} -> {new_name} (Fluid: {fluid_percentage:.2f}%)")

# Usage
if __name__ == "__main__":
    data_folder = "./data"  # Replace with your data folder path if different
    categorize_images(data_folder)
