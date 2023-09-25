import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from utils.speech_command import SpeechCommand
from PIL import Image
# import numpy as np
# import torchvision.transforms.functional as F
# import pdb


# get dataset for training and testing
def get_dataset(args, config):
    if config.data.random_flip is False:
        print("random_filp is False")
        tran_transform = test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )
    else:
        print("random_filp is true")
        tran_transform = transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )

    # path for training dataset, will create path named data/ under current directory
    folder_path = './data'

    if config.data.dataset == "CIFAR10":
        dataset = CIFAR10(
            os.path.join(folder_path, "datasets", "cifar10"),
            train=True,
            download=True,
            transform=tran_transform,
        )
        test_dataset = CIFAR10(
            os.path.join(folder_path, "datasets", "cifar10_test"),
            train=False,
            download=True,
            transform=test_transform,
        )
    
    # loading speech command mel spectrogarm dataset 
    # SpeechCommand Class is written in utils/speech_command.py
    elif config.data.dataset == "SpeechCommand":
        dataset = SpeechCommand(
            # os.path.join(folder_path, "datasets", "speech_command_train_64"),
            os.path.join(folder_path, config.data.dataset, "train"),
            transform=transforms.Compose(
                [
                    transforms.Resize(config.data.image_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                ]
            ),
        )
        test_dataset = SpeechCommand(
            # os.path.join(folder_path, "datasets", "speech_command_test_64"),
            os.path.join(folder_path, config.data.dataset, "test"),
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                ]
            ),
        )
    else:
        # if not yml config passed in, then no dataset will be assigned
        dataset, test_dataset = None, None

    return dataset, test_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)


# get the out of distribution target dataset
def get_targetset(dataset_name, args, config):
    if config.data.random_flip is False:
        tran_transform = test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )
    else:
        tran_transform = transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )

    folder_path = './data'

    if dataset_name == "CIFAR10":
        dataset = CIFAR10(
            os.path.join(folder_path, "datasets", "cifar10"),
            train=True,
            download=True,
            transform=tran_transform,
        )
        test_dataset = CIFAR10(
            os.path.join(folder_path, "datasets", "cifar10_test"),
            train=False,
            download=True,
            transform=test_transform,
        )


    elif dataset_name == "out_class":

        # out of class target class/distribution
        # data_path = os.path.join(folder_path, 'sc', 'train', str(args.target_label))
        # test_data_path = os.path.join(folder_path, 'sc', 'test', str(args.target_label))
        data_path = os.path.join(folder_path, dataset_name, 'train')
        test_data_path = os.path.join(folder_path, dataset_name, 'test')

        data_names = os.listdir(data_path)
        dataset = list()
        label = args.target_label
        for i in range(len(data_names)):
            tmp_path = os.path.join(data_path, data_names[i])
            img = Image.open(tmp_path).convert('RGB')
            img = tran_transform(img)
            dataset.append((img, label))


        data_names = os.listdir(test_data_path)
        test_dataset = list()
        for i in range(len(data_names)):
            tmp_path = os.path.join(test_data_path, data_names[i])
            img = Image.open(tmp_path).convert('RGB')
            img = test_transform(img)
            test_dataset.append((img, label))


    elif dataset_name == "MNIST":
        # # target distribution
        # dataset = MNIST(
        #     os.path.join(folder_path, "datasets", "mnist"),
        #     train=True,
        #     download=True,
        #     transform=tran_transform,
        # )
        # test_dataset = MNIST(
        #     os.path.join(folder_path, "datasets", "mnist_test"),
        #     train=False,
        #     download=True,
        #     transform=test_transform,
        # )

        # target class
        data_path = os.path.join(folder_path, 'mnist', 'train', str(args.target_label))
        test_data_path = os.path.join(folder_path, 'mnist', 'test', str(args.target_label))

        data_names = os.listdir(data_path)
        dataset = list()
        label = args.target_label
        for i in range(len(data_names)):
            tmp_path = os.path.join(data_path, data_names[i])
            img = Image.open(tmp_path).convert('RGB')
            img = tran_transform(img)
            dataset.append((img, label))


        data_names = os.listdir(test_data_path)
        test_dataset = list()
        for i in range(len(data_names)):
            tmp_path = os.path.join(test_data_path, data_names[i])
            img = Image.open(tmp_path).convert('RGB')
            img = test_transform(img)
            test_dataset.append((img, label))

    else:
        dataset, test_dataset = None, None

    return dataset, test_dataset


