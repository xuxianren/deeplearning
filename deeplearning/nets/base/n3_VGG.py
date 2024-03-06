from torch import nn


def vgg_block(n_convs, in_channels, out_channels):
    layers = []
    for _ in range(n_convs):
        layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        ]
        in_channels = out_channels
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        conv_arch = (
            (1, 64),
            (1, 128),
            (2, 256),
            (2, 512),
            (2, 512),
        )

        conv_blks = []
        in_channels = 1
        for n_convs, out_channels in conv_arch:
            conv_blks.append(vgg_block(n_convs, in_channels, out_channels))
            in_channels = out_channels

        self.net = nn.Sequential(
            *conv_blks,
            nn.Flatten(),
            nn.Linear(out_channels * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),
        )
