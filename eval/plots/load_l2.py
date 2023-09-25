

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_multiple_L2_distances(file_paths, labels, colors, zoom_range=None, inset_size="50%"):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for file_path, label, color in zip(file_paths, labels, colors):
        l2_distances = np.load(file_path)
        ax.plot(l2_distances, label=label, color=color)
    
    ax.set_xlabel('Denoise Timesteps', fontsize=28)
    ax.set_ylabel('L2 Distance', fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    ax.legend(fontsize=22, loc="upper right")
    
    if zoom_range:
        x1, x2, y1, y2 = zoom_range
        axins = inset_axes(ax, width=inset_size, height=inset_size, loc='lower left')
        
        for file_path, label, color in zip(file_paths, labels, colors):
            l2_distances = np.load(file_path)
            axins.plot(l2_distances, color=color)
        
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticklabels('')
        axins.set_yticklabels('')
        
        ax.indicate_inset_zoom(axins)
    
    plt.tight_layout()
    plt.show()


# Usage
file_paths = ['./l2_distances_bd_infrasound.npy', './l2_distances_bd_esc50_noise.npy', './l2_distances_bd_concat_gaussian_noise.npy', './l2_distances_bd_gaussian_noise.npy', './l2_distances_bd_patch_white.npy', './l2_distances_bd_hello_kitty.npy']
labels = ['Infrasound', 'Envir.', 'Geometric Noise', 'Gaussian Noise', 'Patch White', 'Hello Kitty']
colors = ['r', 'g', 'm', 'c', 'b', '#A52A2A']

clean_path = ["./l2_distances_clean.npy"]
clean_labels = ["CLean"]
color = ["k"]

zoom_range = (940, 1000, 0.0, 0.6)  # Adjust this range for further zoom
inset_size = "42%"  # Adjust this for larger inset
plot_multiple_L2_distances(file_paths, labels, colors, zoom_range=zoom_range, inset_size=inset_size)
plot_multiple_L2_distances(clean_path, clean_labels, color)












# def plot_L2_distances(file_path, title):
#     l2_distances = np.load(file_path)
#     
#     fig_width, fig_height = 10, 5
#     plt.figure(figsize=(fig_width, fig_height))
#     
#     # Calculate proportional font size
#     prop_fontsize = int((fig_width * fig_height) * 0.5)
#     
#     plt.plot(l2_distances)
#     plt.xlabel('Consecutive Images', fontsize=prop_fontsize)
#     plt.ylabel('L2 Distance', fontsize=prop_fontsize)
#     plt.xticks(fontsize=prop_fontsize)
#     plt.yticks(fontsize=prop_fontsize)
#     
#     plt.tight_layout()  # Adjust layout to prevent clipping
#     plt.show()

# # 
# plot_L2_distances('./l2_distances_1000.npy', 'L2 Distance between Consecutive Timesteps (Clean)')
# plot_L2_distances('./l2_distances_bd_1000.npy', 'L2 Distance between Consecutive Timesteps (backdoor)')

