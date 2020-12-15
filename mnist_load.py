import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid
import numpy as np

"""
returns a 2 class version of MNIST with one class randomly rotated.

TODO: decorate with extended class label.
TODO: rotate digits.
"""
def mnist_data(
        class0=3,
        class1=4,
        rotateclass=1,
        batch_size=64,
        train=True):
    dataset = datasets.MNIST(
        root="./data",
        download=True,
        train=train,
        transform=transform)
    ourclasses = [
        i for i, (x,y) in enumerate(dataset)
        if y in (class0, class1)]
    filt_datset = Subset(dataset, ourclasses)
    data_loader = DataLoader(
        dataset=filt_datset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    return data_loader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5), std=(0.5)),
])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5), std=(0.5)),
])


