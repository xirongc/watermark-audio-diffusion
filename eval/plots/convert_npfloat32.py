

import os
import cv2
import numpy as np

def convert_images_to_float32(src_folder, dest_folder):
    # create direcotry
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # get images file list
    image_files = [f for f in os.listdir(src_folder) if f.endswith('.png') or f.endswith('.jpg')]

    for image_file in image_files:
        # Construct full path
        src_path = os.path.join(src_folder, image_file)
        
        # Read image in grayscale
        img_gray = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)

        # Convert image to np.float32
        img_float32 = np.float32(img_gray) / 255.0

        # Save the converted image to the destination folder
        dest_path = os.path.join(dest_folder, image_file)
        cv2.imwrite(dest_path, img_float32 * 255)

        print(f"Processed and saved: {dest_path}")

convert_images_to_float32('./1000/', './float32_1000')





