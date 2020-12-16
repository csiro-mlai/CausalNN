import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.utils import make_grid
import numpy as np
import random

"""
Returns a 2 class version of MNIST with one class randomly rotated.

TODO: decorate with extended class label.
TODO: rotate digits.
"""

class TwoClassMNIST(Dataset):
    r"""
    Filter MNIST to two classes.
    Transform class1 with 50% prob.
    label is augmented to (digit, rotated)

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(
            self, dataset,
            class0=3,
            class1=4,
            rotateclass=1):
        self.dataset = dataset
        self.transform = transforms.RandomRotation(90)
        self.indices0 = [
            i for i, (x,y) in enumerate(dataset)
            if y == class0]
        self.indices1 = [
            i for i, (x,y) in enumerate(dataset)
            if y == class1]
        self.indices = self.indices0 + self.indices1

    def __getitem__(self, idx):
        im, label = self.dataset[self.indices[idx]]
        rotate = random.choice((0,1)) 
        print(repr(labels), repr(rotate))
        return self.transform(im), (label, rotate)

    def __len__(self):
        return len(self.indices)


def mnist_data(
        batch_size=64,
        train=True):
    
    data_loader = DataLoader(
        dataset=filt_datset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    return data_loader

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5), std=(0.5)),
])
