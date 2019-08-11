import torch
import torch.nn as nn
import torch.nn.functional as F

from . import MLP
from .. import functional as pwF


class DynamicSelfAttentionEncoder(nn.Module):
    """
    Dynamic Self Attention Encoder (https://arxiv.org/abs/1808.07383).
    """

    def __init__(self,
                 time_step_size,
                 att_scores_nb=1,
                 att_iterations=2,
                 projection_size=100,
                 projection_activation=nn.LeakyReLU,
                 attended_representation_activation=nn.Tanh,
                 is_end_padded=True):
        """
        :param time_step_size: Time step size.
        :param att_scores_nb: Number of attended representations.
        :param att_iterations: Number of iterations of the dynamic self-attention algorithm.
        :param projection_size: Size of the projection layer.
        :param projection_activation: Callable that creates the activation of the projection layer.
        :param attended_representation_activation: Callable that creates the activation used on the attended
            representations after each iteration.
        :param is_end_padded: Whether to mask at the end.
        """

        super(DynamicSelfAttentionEncoder, self).__init__()

        self._att_scores_nb = att_scores_nb
        self._att_iterations = att_iterations
        self._projection_size = projection_size
        self._projection_layer = MLP(time_step_size,
                                     num_hidden_layers=0,
                                     output_size=att_scores_nb * projection_size,
                                     output_activation=projection_activation)

        self._attended_representation_activation = attended_representation_activation()
        self._is_end_padded = is_end_padded

    def forward(self, batch_sequences, batch_sequence_lengths):
        """
        :param batch_sequences: 3D Tensor (batch_size, sequence_length, time_step_size).
        :param batch_sequence_lengths: 1D Tensor (batch_size) containing the lengths of the sequences.
        :return: 2D Tensor (batch_size, projection_size * att_scores_nb) containing the encodings.
        """
        projected_batch_sequences = self._projection_layer(batch_sequences).view(batch_sequences.shape[0],
                                                                                 batch_sequences.shape[1],
                                                                                 self._att_scores_nb,
                                                                                 self._projection_size)

        att_scores_matrix = torch.zeros(list(batch_sequences.shape[:-1]) + [self._att_scores_nb],
                                        device=batch_sequences.device)

        mask = pwF.create_mask_from_length(length_tensor=batch_sequence_lengths, mask_size=batch_sequences.shape[1],
                                           zeros_at_end=self._is_end_padded)
        mask = mask.unsqueeze(-1).expand(batch_sequences.shape[0], batch_sequences.shape[1], self._att_scores_nb)

        for it in range(self._att_iterations):
            att_scores_matrix_sm = att_scores_matrix.masked_fill(mask == 0, -1e9)
            att_scores_matrix_sm = F.softmax(att_scores_matrix_sm, dim=-2).unsqueeze(-1)
            z = self._attended_representation_activation(
                torch.sum(projected_batch_sequences * att_scores_matrix_sm, dim=1, keepdim=True))

            if it < self._att_iterations - 1:
                z = z.expand(batch_sequences.shape[0],
                             batch_sequences.shape[1],
                             self._att_scores_nb,
                             self._projection_size)
                att_scores_matrix += torch.sum(z * projected_batch_sequences, dim=-1, keepdim=False)

        return z.view(batch_sequences.shape[0], -1)
