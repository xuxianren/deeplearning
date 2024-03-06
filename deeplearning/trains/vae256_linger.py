import torch
from torch import nn
import torch.utils.data
import torchvision.datasets as dset
from torchvision import transforms
from dataclasses import dataclass

from gm.models.vae import VAE
from gm.utils import show_imgs


@dataclass
class Hparams:
    dataroot = "/data/dataset/linger256"
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
ndf = 32
nz = ndf * 32
ngf = 32


def train(dataloader, model, loss_fn, optimizer, device="cuda"):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        x_hat, mean, variance = model(X)
        loss = loss_fn(x_hat, X, mean, variance)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        # if batch == 0:
        #     show_imgs(x_hat)


vae = VAE()
vae.to(device)

# loss_fn = nn.MSELoss()


def loss_fn(x_hat, x, mean, variance):
    l2_loss = nn.functional.mse_loss(x_hat, x)
    # loss2 = torch.sum(torch.exp(variance) - (1 + variance) + torch.sqrt(mean))
    # print(loss2)
    kl_loss = -0.5 * torch.sum(1 + variance - torch.sqrt(mean) - torch.exp(variance))

    # loss = 0.95 * l2_loss + 0.05 * kl_loss
    loss = l2_loss + 0.001 * kl_loss
    return loss


optimizer = torch.optim.Adam(vae.parameters())

if __name__ == "__main__":
    import os

    name = os.path.splitext(os.path.basename(__file__))[0]
    epoch_nums = 10
    for epoch in range(epoch_nums):
        train(dataloader, vae, loss_fn, optimizer)
        torch.save(
            vae.state_dict(),
            f"D:\projects\learning\gandemo\checkpoints\\{name}_{epoch}.pth",
        )

    # def test_decoder():
    #     model_path = (
    #         f"D:\projects\learning\gandemo\checkpoints\\{name}_{epoch_nums-1}.pth"
    #     )
    #     vae.load_state_dict(torch.load(model_path))
    #     noise = torch.randn(8, nz, 1, 1, dtype=torch.float32)
    #     noise = noise.to(device)
    #     fake = vae.decoder(noise)
    #     show_imgs(fake)

    # test_decoder()
