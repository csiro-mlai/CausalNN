import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np


def causal_controller():
    pass


def discriminator(features, channels=1):
    D = nn.Sequential(
        # Batch_size x image_channels x 32 x 32
        nn.Conv2d(channels, features,
                  kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.5),
        # Batch_size x features  x 16 x 16
        nn.Conv2d(features, features * 2,
                  kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.4),
        # Batch_size x (features * 2) x 8 x 8
        nn.Conv2d(features * 2, features * 4,
                  kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.4),
        # Batch_size x (features * 4) x 4 x 4
        nn.Conv2d(features * 4, 1,
                  kernel_size=4, stride=2, padding=0),
        # Batch x 1 x 1 x 1
        nn.Sigmoid(),
    )
    return D


def generator(latent_size, features, channels=1):
    G = nn.Sequential(
        # Batch_size x noise_features x 1 x 1
        nn.ConvTranspose2d(latent_size, features * 4,
                           kernel_size=4, stride=1, padding=0),
        nn.BatchNorm2d(features * 4),
        nn.LeakyReLU(0.2),
        # Batch x  (features * 4) x 4 x 4
        nn.ConvTranspose2d(features * 4, features * 2,
                          kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(features * 2),
        nn.LeakyReLU(0.2),
        # Batch x (features * 2) x 8 x 8
        nn.ConvTranspose2d(features * 2, features,
                          kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(features),
        nn.LeakyReLU(0.2),
        # Batch x (features) x 16 x 16
        nn.ConvTranspose2d(features, channels,
                           kernel_size=4, stride=2, padding=1),
        # Batch x image_channels X 32 X 32
        nn.Tanh()
    )

    return G


def labler(features, channels=1):
    L = nn.Sequential(
        # Batch_size x image_channels x 32 x 32
        nn.Conv2d(channels, features,
                  kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.5),
        # Batch_size x features  x 16 x 16
        nn.Conv2d(features, features * 2,
                  kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.4),
        # Batch_size x (features * 2) x 8 x 8
        nn.Conv2d(features * 2, features * 4,
                  kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.4),
        # Batch_size x (features * 4) x 4 x 4
        nn.Conv2d(features * 4, 1,
                  kernel_size=4, stride=2, padding=0),
        # Batch x 1 x 1 x 1
        nn.Sigmoid(),
    )
    return L


def anti_labler(features, channels=1):
    AL = nn.Sequential(
        # Batch_size x image_channels x 32 x 32
        nn.Conv2d(channels, features,
                  kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.5),
        # Batch_size x features  x 16 x 16
        nn.Conv2d(features, features * 2,
                  kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.4),
        # Batch_size x (features * 2) x 8 x 8
        nn.Conv2d(features * 2, features * 4,
                  kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.4),
        # Batch_size x (features * 4) x 4 x 4
        nn.Conv2d(features * 4, 1,
                  kernel_size=4, stride=2, padding=0),
        # Batch x 1 x 1 x 1
        nn.Sigmoid(),
    )
    return AL