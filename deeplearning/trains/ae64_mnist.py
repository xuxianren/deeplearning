import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from deeplearning.nets.autoencoders.ae import AutoEncoder64


device = "cuda"

training_data = datasets.MNIST(
    root="/data/dataset",
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Resize(64),
            # transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    ),
)

# test_data = datasets.MNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=transforms.ToTensor(),
# )

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=batch_size)


def train(dataloader, model, loss_fn, optimizer, device="cuda"):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        x_hat = model(X)
        loss = loss_fn(x_hat, X)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        if batch == 0:
            show_imgs(x_hat)


auto_encoder = AutoEncoder64(nc=1, ndf=32)
auto_encoder.to(device)


loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(auto_encoder.parameters())

for epoch in range(10):
    train(train_dataloader, auto_encoder, loss_fn, optimizer)
