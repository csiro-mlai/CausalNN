import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import pathlib

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
    data_loader = DataLoader(
        dataset=dataset,
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


