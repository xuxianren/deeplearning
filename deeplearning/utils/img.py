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
                batch.to("cpu")[:256],
                padding=16,
                normalize=True,
            ).cpu(),
            (1, 2, 0),
        )
    )
    plt.show()
