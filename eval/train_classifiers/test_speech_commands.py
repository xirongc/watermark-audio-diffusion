#!/usr/bin/env python
"""Test a pretrained CNN for Google speech commands."""

__author__ = 'Yuan Xu, Erdene-Ochir Tuguldur'

import argparse
import time
import csv
import os

from tqdm import *

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision.transforms import *
import torchnet

from datasets import *


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--test_dataset", type=str, default='./data/test', help='path of test dataset')
parser.add_argument("--batch_size", type=int, default=32, help='batch size')
parser.add_argument("--dataload_workers_nums", type=int, default=4, help='number of workers for dataloader')
parser.add_argument("--input", choices=['mel32'], default='mel32', help='input of NN')
parser.add_argument('--output', type=str, default='', help='save output to file for the kaggle competition, if empty the model name will be used')
#parser.add_argument('--prob-output', type=str, help='save probabilities to file', default='probabilities.json')
parser.add_argument("--model", help='a pretrained neural network model')
args = parser.parse_args()

dataset_dir = args.test_dataset

print("loading model...")
model = torch.load(args.model)
# model.float()

use_gpu = torch.cuda.is_available()
print('use_gpu', use_gpu)
if use_gpu:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    model.cuda()

test_dataset = SpeechCommandsDataset(dataset_dir, transforms.ToTensor())
# test_dataset = BackdoorDataset(dataset_dir, transforms.ToTensor(), backdoor_cls=6)

test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=None,
                            pin_memory=use_gpu, num_workers=args.dataload_workers_nums)

criterion = torch.nn.CrossEntropyLoss()

def test():
    model.eval()  # Set model to evaluate mode

    #running_loss = 0.0
    #it = 0
    correct = 0
    total = 0
    # confusion_matrix = torchnet.meter.ConfusionMeter(len(USED_CLS))
    predictions = {}
    probabilities = {}

    pbar = tqdm(test_dataloader, unit="audios", unit_scale=test_dataloader.batch_size)
    for batch in pbar:
        inputs = batch['input']
        # inputs = torch.unsqueeze(inputs, 1)
        targets = batch['target']

        n = inputs.size(0)

        inputs = Variable(inputs)
        targets = Variable(targets, requires_grad=False)

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda(non_blocking=True)

        # forward
        outputs = model(inputs)
        #loss = criterion(outputs, targets)
        outputs = torch.nn.functional.softmax(outputs, dim=1)

        # statistics
        #it += 1
        #running_loss += loss.data[0]
        pred = outputs.data.max(1, keepdim=True)[1]
        correct += pred.eq(targets.data.view_as(pred)).sum()
        total += targets.size(0)
        
        # print("pred.size: {}".format(pred.shape))
        # print("targets.size: {}".format(targets.data.shape))
        
        # confusion_matrix.add(pred, targets.data.unsqueeze(-1))

        filenames = batch['path']
        for j in range(len(pred)):
            fn = filenames[j]
            predictions[fn] = pred[j][0]
            probabilities[fn] = outputs.data[j].tolist()

    accuracy = correct/total
    #epoch_loss = running_loss / it
    print("accuracy: %f%%" % (100*accuracy))
    # print("confusion matrix:")
    # print(confusion_matrix.value())

    return probabilities, predictions

print("testing...")
probabilities, predictions = test()
