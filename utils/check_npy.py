
import numpy as np

# Load the data and label for a specific split (e.g., 'train')
data_path = "./raw/npy/val_data.npy"
label_path = "./raw/npy/val_label.npy"

# Load the .npy files
loaded_data = np.load(data_path)
loaded_labels = np.load(label_path)

def print_structure(data, labels):
    print("Dataset Structure:")
    print(f"  ├─ Data shape: {data.shape}")
    print(f"  │    └─ Sample data: {data[0][:5]}... (total {data[0].shape[0]} samples)")
    print(f"  └─ Labels shape: {labels.shape}")
    print(f"       └─ Sample label: {labels[0]}")


# Print the structure
print_structure(loaded_data, loaded_labels)

