import torch
import torch.utils.data
import torchvision.datasets as dset
from torch import nn
from torchvision import transforms
from dataclasses import dataclass
from torchsummary import summary


@dataclass
class Hparams:
    dataroot = r"D:\projects\learning\gandemo\data\image256"
    workers = 1
    batch_size = 8
    image_size = 256
    ngpu = 1


device = torch.device(
    "cuda:0" if (torch.cuda.is_available() and Hparams.ngpu > 0) else "cpu"
)


dataset = dset.ImageFolder(
    root=Hparams.dataroot,
    transform=transforms.Compose(
        [
            # transforms.Resize(Hparams.image_size),
            # transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
)
# Create the dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=Hparams.batch_size,
    shuffle=True,
    num_workers=Hparams.workers,
)

nc = 3
ndf = 64
nz = ndf * 32
ngf = 64


class AutoEncoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = nn.Sequential(
            # 256 -> 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 -> 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # raw
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 32, ndf * 32, 4, 1, 0, bias=False),
        )
        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 32),
            nn.ReLU(True),
            #
            nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            #
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        self.net = nn.Sequential(
            self.encoder,
            self.decoder,
        )

    def forward(self, x):
        return self.net(x)


def show_imgs(batch):
    import matplotlib.pyplot as plt
    import torchvision.utils as vutils
    import numpy as np

    plt.figure(figsize=(2, 4))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                batch.to(device)[:256],
                padding=16,
                normalize=True,
            ).cpu(),
            (1, 2, 0),
        )
    )
    plt.show()


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
        # if batch == 0:
        #     show_imgs(x_hat)


auto_encoder = AutoEncoder()
auto_encoder.to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(auto_encoder.parameters())

if __name__ == "__main__":
    # for epoch in range(10):
    #     train(dataloader, auto_encoder, loss_fn, optimizer)
    #     torch.save(
    #         auto_encoder.state_dict(),
    #         f"D:\projects\learning\gandemo\checkpoints\\auto_encoder_{epoch}.pth",
    #     )

    def test_decoder():
        model_path = r"D:\projects\learning\gandemo\checkpoints\auto_encoder_9.pth"
        auto_encoder.load_state_dict(torch.load(model_path))
        noise = torch.randn(8, nz, 1, 1, dtype=torch.float32)
        noise = noise.to(device)
        fake = auto_encoder.decoder(noise)
        show_imgs(fake)

    test_decoder()
