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


def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()
    l_optimizer.zero_grad()
    al_optimizer.zero_grad()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_size = 10
    latent_size = 64
    hidden_size = 256
    image_size = 784
    num_epochs = 20
    batch_size = 64
    p = 0.5
    lr = 0.0002
    print_interval = 400
    show_interval = num_epochs // 5
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5), std=(0.5)),
                                    ])
    dataset = datasets.MNIST(root="./data", download=True, transform=transform)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    D = discriminator(image_size, hidden_size).to(device)
    G = generator(latent_size + label_size, hidden_size).to(device)
    L = labler(image_size, hidden_size).to(device)
    AL = anti_labler(image_size, hidden_size).to(device)

    d_optimizer = torch.optim.Adam(D.parameters(), lr=lr)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr)
    l_optimizer = torch.optim.Adam(L.parameters(), lr=lr)
    al_optimizer = torch.optim.Adam(AL.parameters(), lr=lr)

    total_step = len(data_loader)

    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    class_1_label = [0, 1]
    class_0_label = [1, 0]

    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(data_loader):
            images = images.reshape(batch_size, -1).to(device)
            images_positive =
            images_negative =

            z = torch.randn(batch_size, latent_size).to(device)
            concat_zl_positive = torch.cat((z, class_1_label),
                                           -1)  # I suppose the consitioning label is concatenated to z as in CVAE?
            concat_zl_negative = torch.cat((z, class_0_label), -1)

            fake_images_positive = G(concat_zl_positive)
            fake_images_negative = G(concat_zl_negative)
            fake_images_total = torch.cat((fake_images_positive, fake_images_negative), 0)

            # train discriminator
            # real images
            d_real_outputs = D(images)
            d_loss_real = torch.log(d_real_outputs)
            d_loss_real = torch.mean(d_loss_real)
            real_score = d_real_outputs.mean().item()
            # fake images
            d_fake_outputs = D(fake_images_total)
            d_loss_fake = torch.log(1 - d_outputs / (d_outputs + 1e-3))
            d_loss_fake = torch.mean(d_loss_fake)
            fake_score = d_fake_outputs.mean().item()
            # loss
            d_loss = -d_loss_real - d_loss_fake
            reset_grad()
            d_loss.backward()
            d_optimizer.step()

            # train labler
            positive_outputs = L(images_positive)
            negative_outputs = L(images_negative)
            l_loss = p * torch.log(positive_outputs) + (1 - p) * torch.log(1 - negative_outputs)
            l_loss = -1 * torch.mean(l_loss)
            reset_grad()
            l_loss.backward()
            l_optimizer.step()

            # train anti-labler
            positive_outputs = AL(fake_images_positive)
            negative_outputs = AL(fake_images_negative)
            al_loss = p * torch.log(positive_outputs) + (1 - p) * torch.log(1 - negative_outputs)
            al_loss = -1 * torch.mean(al_loss)
            reset_grad()
            al_loss.backward()
            al_optimizer.step()

            # train generator
            d_real_outputs = D(images)  # recalculate the outputs since D, L, and AL are changed.
            d_fake_outputs = D(fake_images_total)
            positive_outputs = L(images_positive)
            negative_outputs = L(images_negative)
            positive_outputs = AL(fake_images_positive)
            negative_outputs = AL(fake_images_negative)

            term_1 = p * torch.log(positive_outputs) + (1 - p) * torch.log(1 - negative_outputs)
            term_2 = - p * torch.log(positive_outputs) - (1 - p) * torch.log(1 - negative_outputs)
            term_3 = p * torch.log(positive_outputs) + (1 - p) * torch.log(1 - negative_outputs)
            g_loss = term_1 + term_2 + term_3
            reset_grad()
            g_loss.backward()
            g_optimizer.step()

            if (i + 1) % print_interval == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Step [{i + 1}/{total_step}], "
                    f"d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, "
                    f"D(x): {real_score:.2f}, D(G(z)): {fake_score:.2f}"
                )
        if (epoch + 1) % show_interval == 0:
            fake_images = fake_images.reshape(batch_size, 1, 28, 28).detach()
            plt.imshow(make_grid(fake_images).permute(1, 2, 0))
            plt.axis("off")
            plt.show()