

from PIL import Image
import os
import argparse

# Function to check the format of all images in a directory and its subdirectories
def check_image_formats(directory_path):
    dir_formats = {}
    
    for root, _, files in os.walk(directory_path):
        local_formats = {}
        for filename in files:
            if filename.endswith(('.png', '.jpg', '.jpeg')):  # Add other image formats if needed
                img_path = os.path.join(root, filename)
                img = Image.open(img_path)
                img_format = img.mode  # 'RGB' for color images, 'L' for grayscale, etc.
                
                if img_format not in local_formats:
                    local_formats[img_format] = 0
                local_formats[img_format] += 1
        
        if local_formats:  # Only print if there are any images
            print(f"{root}: {local_formats}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check image formats in a directory.')
    parser.add_argument('--dir', type=str, required=True, help='Path to the directory containing images.')
    
    args = parser.parse_args()
    check_image_formats(args.dir)

