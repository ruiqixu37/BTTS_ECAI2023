import torch
import torch.nn as nn


class BarlowTwinsLoss(nn.Module):
    def __init__(self, batch_size=64, lambda_coeff=5e-3, z_dim=128):
        super().__init__()

        self.z_dim = z_dim
        self.batch_size = batch_size
        self.lambda_coeff = lambda_coeff

    def off_diagonal_ele(self, x):
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        z1 = z1.flatten(start_dim=1)
        z2 = z2.flatten(start_dim=1)

        # N x D, where N is the batch size and D is output dim of projection head
        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)

        cross_corr = torch.matmul(z1_norm.T, z2_norm) / self.batch_size

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        return on_diag + self.lambda_coeff * off_diag


class TripletLoss(nn.Module):
    def __init__(self, batch_size=64, lambda_coeff=1, z_dim=128):
        super().__init__()

        self.z_dim = z_dim
        self.batch_size = batch_size
        self.lambda_coeff = lambda_coeff

    def forward(self, z1, z2, z3):
        '''
        z2 is positive sample and z3 is negative sample
        '''
        z1 = z1.flatten(start_dim=1)
        z2 = z2.flatten(start_dim=1)
        z3 = z2.flatten(start_dim=1)

        loss = torch.square(z1 - z2) - torch.square(z1 -
                                                    z3) + self.lambda_coeff

        return torch.max(loss, 0)
