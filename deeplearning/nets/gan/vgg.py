import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from generate.utils import train, test


def vgg_block(n_conv, c_in, c_out):
    layers = []
    for _ in range(n_conv):
        layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        c_in = c_out
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class Discriminator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(
            vgg_block(1, 1, 16),  # 28 -> 14
            vgg_block(1, 16, 32),  # 14 -> 7
            vgg_block(2, 32, 64),  # 7->3
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential()

    def forward(self, x):
        return self.net(x)


training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


def train_Discriminator():
    model = Discriminator()
    model.to("cuda")
    optmizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epoch = 10
    loss_fn = nn.CrossEntropyLoss()
    for n in range(epoch):
        train(
            train_dataloader,
            model,
            loss_fn,
            optmizer,
        )
        test(test_dataloader, model, loss_fn)


if __name__ == "__main__":
    train_Discriminator()
