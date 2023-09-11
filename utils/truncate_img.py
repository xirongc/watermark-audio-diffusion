

import os
import argparse
from tqdm import tqdm
import random

def truncate_every_directory_to_certain_count(original_folder, num_images):
    """
    Truncate the original dataset to contain only 'num_images' in each subdirectory.
    """
    
    # Count total number of image files across all subdirectories
    total_files = 0
    for root, _, files in os.walk(original_folder):
        total_files += len([f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Choose between tqdm and print statements based on total_files
    if total_files > 100000:
        with tqdm(total=total_files, unit='file', bar_format='{l_bar}{bar}{r_bar}', ascii='->=') as t:
            for root, _, files in os.walk(original_folder):
                image_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]
                files_to_keep = random.sample(image_files, min(num_images, len(image_files)))
                files_to_delete = set(image_files) - set(files_to_keep)
                for file in files_to_delete:
                    os.remove(os.path.join(root, file))
                    t.update(1)
    else:
        print("About to truncate...")
        for root, _, files in os.walk(original_folder):
            image_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]
            files_to_keep = random.sample(image_files, min(num_images, len(image_files)))
            files_to_delete = set(image_files) - set(files_to_keep)
            for file in files_to_delete:
                os.remove(os.path.join(root, file))
        print("Truncation complete.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Truncate image dataset.')
    parser.add_argument('--dir', type=str, required=True, help='Original directory to truncate.')
    parser.add_argument('--truncate_num', type=int, required=True, help='Number of images to keep in each subdirectory.')
    
    args = parser.parse_args()
    truncate_every_directory_to_certain_count(args.dir, args.truncate_num)
