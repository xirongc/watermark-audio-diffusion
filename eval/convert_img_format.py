
import os
import argparse
from PIL import Image
from tqdm import tqdm

def convert_images(directory, to_rgb):
    # Count total number of image files across all subdirectories
    total_files = 0
    for root, _, files in os.walk(directory):
        total_files += len([f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Choose between tqdm and print statements based on total_files
    if total_files > 10000:
        print(f"Starting image conversion...RGB: {to_rgb}")
        with tqdm(total=total_files, unit='file', bar_format='{l_bar}{bar}{r_bar}', ascii='->=') as t:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(root, file)
                        img = Image.open(file_path)
                        if to_rgb and img.mode != 'RGB':
                            img = img.convert('RGB')
                        elif not to_rgb and img.mode != 'L':
                            img = img.convert('L')
                        img.save(file_path)
                        t.update(1)
        print("Image conversion complete.")

    else:
        print("Starting image conversion...")
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file)
                    img = Image.open(file_path)
                    if to_rgb and img.mode != 'RGB':
                        img = img.convert('RGB')
                    elif not to_rgb and img.mode != 'L':
                        img = img.convert('L')
                    img.save(file_path)
        print("Image conversion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert image colors.')
    parser.add_argument('--dir', type=str, required=True, help='Directory containing images.')
    parser.add_argument('--rgb', action='store_true', help='Convert to RGB if present, to grayscale if not.')
    
    args = parser.parse_args()
    convert_images(args.dir, args.rgb)


'''

argparse library handles Boolean arguments:
When you pass --rgb False, argparse still considers it as True because the argument is present.

Instead of using type=bool for --rgb:
1) use action='store_true' to set it to True only when the flag is provided. If the flag is not provided, it will default to False(boolean). 
2) specify --rgb when converting to RGB and omit it for grayscale convertion

'''
