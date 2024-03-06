import torch
from torch import nn

class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.relu = nn.ReLu()
    
    def forward(self, x):
        pass


class VQVAE(nn.module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.encoder = nn.Sequential(

        )
        self.decoder = nn.Sequential(

        )
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform(-1.0/num_embeddings, 1.0/num_embeddings)
    
    def forward(self, x):
        ze = self.encoder(x)
        n, c, h, w = ze.shape

        codebook = self.codebook.weight.data
        k, d = codebook.shape
        ze_broadcast = ze.reshape(n, 1, c, h, w)
        embedding_broadcast = codebook.reshape(1, k, c, 1, 1)
        distance = torch.sum((embedding_broadcast - ze_broadcast) ** 2, 2)
        q_idx = torch.argmin(distance, 1)
        zq = self.codebook(q_idx).permute(0, 3, 1, 2)
        
        decoder_in = ze + (zq - ze).detach()
        x_hat = self.decoder(decoder_in)
        return x_hat, ze, zq

if __name__ == "__main__":
    from torchsummary import summary
    k, d = 8196, 128
    codebook = nn.Embedding(k, d )
    codebook.weight.data.uniform(-1.0/k, 1.0/k)
    n, c, h, w = 8, d, 16, 16
    ze = torch.randn(n, c, h, w)
    ze_broadcast = ze.reshape()