import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

ALL_CLS = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
USED_CLS = ['bird', 'cat', 'dog', 'down', 'eight', 'marvin', 'sheila', 'three', 'wow', 'zero']


class SpeechCommandsDataset(Dataset):
    def __init__(self, folder, transform=None, classes=USED_CLS):

        all_classes = [d for d in classes if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')]
        for c in classes:
            assert c in all_classes

        class_to_idx = {classes[i]: i for i in range(len(classes))}
        for c in all_classes:
            if c not in class_to_idx:
                class_to_idx[c] = len(classes) - 1


        data = []
        for c in all_classes:
            d = os.path.join(folder, c)
            target = class_to_idx[c]
            for f in os.listdir(d):
                path = os.path.join(d, f)
                data.append((path, target))    

        self.classes = classes
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, target = self.data[index]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        data = {'input': img, 'target': target, 'path': path}

        return data

    def make_weights_for_balanced_classes(self):
        """adopted from https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3"""
        n_classes = len(self.classes)
        count = np.zeros(n_classes)
        for item in self.data:
            count[item[1]] += 1

        N = float(sum(count))
        weight_per_class = N / count
        weight = np.zeros(len(self))
        for idx, item in enumerate(self.data):
            weight[idx] = weight_per_class[item[1]]
        return weight
    
    
class BackdoorDataset(Dataset):
    def __init__(self, folder, transform=None, classes=USED_CLS, backdoor_cls=None):

        data = []
        target = backdoor_cls
        for f in os.listdir(folder):
            path = os.path.join(folder, f)
            data.append((path, target))
    

        self.classes = classes
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, target = self.data[index]
        img = Image.open(path).convert('L')
        if self.transform is not None:
            img = self.transform(img)

        data = {'input': img, 'target': target, 'path': path}

        return data