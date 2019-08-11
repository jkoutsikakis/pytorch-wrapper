from __future__ import division

import torch
import math

from torch import nn


class SinusoidalPositionalEmbeddingLayer(nn.Module):
    """
    Sinusoidal Positional Embeddings (https://arxiv.org/pdf/1706.03762.pdf).
    """

    def __init__(self, emb_size, pad_at_end=True, init_max_sentence_length=1024):
        """
        :param emb_size: Size of the positional embeddings.
        :param pad_at_end: Whether to pad at the end.
        :param init_max_sentence_length: Initial maximum length of sentence.
        """

        super(SinusoidalPositionalEmbeddingLayer, self).__init__()

        self._emb_size = emb_size
        self._pad_at_end = pad_at_end
        self.register_buffer('embeddings', self.create_embeddings(init_max_sentence_length))

    def create_embeddings(self, num_embeddings):

        embeddings = torch.zeros(num_embeddings, self._emb_size)
        for pos in range(num_embeddings):
            for i in range(0, self._emb_size, 2):
                embeddings[pos, i] = math.sin(pos / (10000 ** (i / self._emb_size)))
            for i in range(1, self._emb_size, 2):
                embeddings[pos, i] = math.cos(pos / (10000 ** (i / self._emb_size)))

        embeddings = torch.cat([torch.zeros(1, self._emb_size), embeddings], dim=0)

        return embeddings

    def forward(self, length_tensor, max_sequence_length):
        """
        :param length_tensor: ND Tensor containing the real lengths.
        :param max_sequence_length: Int that corresponds to the size of (N+1)D dimension.
        :return: (N+2)D Tensor with the positional embeddings.
        """

        max_len = length_tensor.max().item()
        if max_len >= self.embeddings.shape[0]:
            self.embeddings = self.create_embeddings(max_len).to(self.embeddings.device)

        index = torch.arange(1, max_sequence_length + 1, dtype=torch.long, device=length_tensor.device)
        index_shape = [1] * len(length_tensor.shape) + [max_sequence_length]
        index = index.view(index_shape)
        index_expand_shape = list(length_tensor.shape) + [max_sequence_length]
        index = index.expand(index_expand_shape)

        if self._pad_at_end:
            mask = (index <= length_tensor.long().unsqueeze(-1))
        else:
            index = index - max_sequence_length + length_tensor.long().unsqueeze(-1)
            mask = (0 < index)

        index = index.masked_fill(mask == 0, 0)

        new_shape = list(length_tensor.shape) + [max_sequence_length, self._emb_size]

        return self.embeddings.index_select(dim=0, index=index.view(-1)).view(*new_shape)
