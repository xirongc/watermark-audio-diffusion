
import os
import shutil
import argparse

def expand_directory(root_directory, output_directory):
    print("Starting to expand directory...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for root, _, files in os.walk(root_directory):
        for file in files:
            src_path = os.path.join(root, file)
            dest_path = os.path.join(output_directory, file)
            
            # Check if a file with the same name already exists in the output directory
            counter = 1
            original_dest_path = dest_path
            while os.path.exists(dest_path):
                dest_path = f"{original_dest_path.rsplit('.', 1)[0]}_{counter}.{original_dest_path.rsplit('.', 1)[1]}"
                counter += 1
            
            shutil.copy2(src_path, dest_path)
                
    print("Expansion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Expand directory.')
    parser.add_argument('--dir', type=str, required=True, help='Directory to expand.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory where files will be moved.')
    
    args = parser.parse_args()
    expand_directory(args.dir, args.output_dir)

