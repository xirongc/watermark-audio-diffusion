
import os
import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset
# from torchvision import transforms

# class for loading speech command dataset
class SpeechCommand(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super(SpeechCommand, self).__init__(root, transform=transform, target_transform=target_transform)

        self.data = []
        self.targets = []
        self.classes = sorted(os.listdir(root))  # get class names

        # Load all images into memory
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root, class_name)
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                # image = Image.open(image_path).convert('L')  # Convert image to grayscale
                image = Image.open(image_path).convert('RGB')  # originally grayscale image will be loaded as 'RGB'
                # image = Image.open(image_path)  # normal open
                image_np = np.array(image)
                self.data.append(image_np)
                self.targets.append(label)

        # Convert data to numpy array
        self.data = np.array(self.data)

    # necessary, because when using DataLoader to iterate through dataset 
    # it needs to fetch each data pair using __getitem__(), though it won't explicitly call it
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # Convert image from numpy array to PIL image
        img = Image.fromarray(img)

        # if transform is used, for in distribution watermark, it's always used
        if self.transform is not None:
            img = self.transform(img)

        # if target_transform is used
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    # get the length of the whole dataset
    def __len__(self):
        return len(self.data)


# test for labeling
# ======================================================================
# path = "../data/datasets/speech_command_64/"
# my_tran=transforms.Compose(
#                 [
#                     # transforms.Resize(config.data.image_size),
#                     transforms.RandomHorizontalFlip(p=0.5),
#                     transforms.ToTensor(),
#                 ]
#             )
# dataset = SPEECH(path, my_tran)
#
# # 
# label_index = 5
# print(dataset.classes[label_index])
# ======================================================================



