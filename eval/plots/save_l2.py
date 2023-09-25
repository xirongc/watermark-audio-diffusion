import cv2
import numpy as np
import os

def sort_key(filename):
    return int(filename.split('_')[1].split('.png')[0])

def sort_key_img(filename):
    return int(filename[3:].split('.png')[0])




def calculate_L2_distances(image_folder_path):
    files = [f for f in os.listdir(image_folder_path) if f.endswith('.png')]
    files.sort(key=sort_key_img)
    
    l2_distances = []
    
    prev_image = None
    counter = 0


    for filename in files:
        print(f"-> {filename}")
        current_image = cv2.imread(os.path.join(image_folder_path, filename), cv2.IMREAD_GRAYSCALE)

        # convert to np.float32
        current_image = np.float32(current_image) / 255.0
        
        if prev_image is not None:
            l2_distance = np.linalg.norm(prev_image - current_image)

            # save every 5 steps
            # counter += 1 
            # if counter % 10 == 0:
            #     l2_distances.append(l2_distance)
            l2_distances.append(l2_distance)
        
        prev_image = current_image
    
    # convert to numpy array
    l2_distances = np.array(l2_distances)
    
    # normalize the L2 distances
    # min_val = np.min(l2_distances)
    # max_val = np.max(l2_distances)
    # normalized_l2_distances = (l2_distances - min_val) / (max_val - min_val)
    
    np.save(f'/Users/mikiyax/Desktop/ablation_study/generation_step/ddpm/l2_distances_{image_folder_path}', l2_distances)

# 
calculate_L2_distances('clean')
calculate_L2_distances('bd_infrasound')
calculate_L2_distances('bd_esc50_noise')
calculate_L2_distances('bd_concat_gaussian_noise')
calculate_L2_distances('bd_gaussian_noise')
calculate_L2_distances('bd_patch_white')
calculate_L2_distances('bd_hello_kitty')








# def calculate_L2_distances(image_folder_path):
#     files = [f for f in os.listdir(image_folder_path) if f.endswith('.png')]
#     files.sort(key=sort_key_img)
#     
#     l2_distances = []
#     
#     prev_image = None
#     counter = 0


#     for filename in files:
#         print(f"-> {filename}")
#         current_image = cv2.imread(os.path.join(image_folder_path, filename), cv2.IMREAD_GRAYSCALE)
#         
#         if prev_image is not None:
#             l2_distance = np.linalg.norm(prev_image - current_image)

#             # save every 5 steps
#             # counter += 1 
#             # if counter % 5 == 0:
#                 # l2_distances.append(l2_distance)
#             l2_distances.append(l2_distance)
#         
#         prev_image = current_image
#     
#     # convert to numpy array
#     l2_distances = np.array(l2_distances)
#     
#     # normalize the L2 distances
#     min_val = np.min(l2_distances)
#     max_val = np.max(l2_distances)
#     normalized_l2_distances = (l2_distances - min_val) / (max_val - min_val)
#     
#     np.save(f'/Users/mikiyax/Desktop/ablation_study/generation_step/ddpm/l2_distances_{image_folder_path}_normalized', normalized_l2_distances)

# # 
# # calculate_L2_distances('1000')
# calculate_L2_distances('1000')

