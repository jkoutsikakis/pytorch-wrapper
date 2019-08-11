import torch

from torch import nn


class LayerNorm(nn.Module):
    """
    Layer Normalization (https://arxiv.org/pdf/1607.06450.pdf).
    """

    def __init__(self, last_dim_size, eps=1e-6):
        """
        :param last_dim_size: Size of last dimension.
        :param eps: Small number for numerical stability (avoid division by zero).
        """

        super(LayerNorm, self).__init__()

        self._a_2 = nn.Parameter(torch.ones(last_dim_size))
        self._b_2 = nn.Parameter(torch.zeros(last_dim_size))
        self._eps = eps

    def forward(self, x):
        """
        :param x: Tensor to be layer normalized.
        :return: Layer normalized Tensor.
        """

        mean = x.mean(dim=-1, keepdim=True).detach()
        std = x.std(dim=-1, keepdim=True).detach()

        return self._a_2 * (x - mean) / (std + self._eps) + self._b_2
