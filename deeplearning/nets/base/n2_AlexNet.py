from torch import nn


class AlexNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(
            # 224 -> 55 -> 27 -> 13 -> 6 -> 3
            # 原始AlexNet中是 3 * 224 * 224
            # (224 + 2 - 11)//4 + 1 = 54
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(),
            # (54 - 3) // 2 + 1 = 26
            nn.MaxPool2d(kernel_size=3, stride=2),
            # (26 + 2*2 - 5) // 1 + 1 = 26
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            # (26 - 3) // 2 + 1 = 12
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 256 * 12 * 12
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            # (12 - 3) // 2 + 1 = 5
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 256 * 5 * 5
            nn.Flatten(),
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 10),
        )
