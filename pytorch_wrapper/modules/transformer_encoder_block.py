import torch.nn as nn

from . import LayerNorm, MultiHeadAttention
from .. import functional as pwF


class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block (https://arxiv.org/pdf/1706.03762.pdf).
    """

    def __init__(self, time_step_size, heads, out_mlp, dp=0, is_end_padded=True):
        """
        :param time_step_size: Time step size.
        :param heads: Number of attention heads.
        :param out_mlp: MLP that will be performed after the attended sequence is generated.
        :param dp: Dropout probability.
        :param is_end_padded: Whether to mask at the end.
        """
        super(TransformerEncoderBlock, self).__init__()

        self._norm_1 = LayerNorm(time_step_size)
        self._norm_2 = LayerNorm(time_step_size)
        self._attn = MultiHeadAttention(time_step_size, heads, dp, is_end_padded)
        self._out_mlp = out_mlp
        self._dropout_1 = nn.Dropout(dp)
        self._dropout_2 = nn.Dropout(dp)
        self._is_end_padded = is_end_padded

    def forward(self, batch_sequences, batch_sequence_lengths):
        """
        :param batch_sequences: batch_sequences: 3D Tensor (batch_size, sequence_length, time_step_size).
        :param batch_sequence_lengths: 1D Tensor (batch_size) containing the lengths of the sequences.
        :return: 3D Tensor (batch_size, sequence_length, time_step_size).
        """

        batch_sequences = batch_sequences + self._attn(batch_sequences,
                                                       batch_sequences,
                                                       batch_sequences,
                                                       batch_sequence_lengths,
                                                       batch_sequence_lengths)

        batch_sequences = self._norm_1(self._dropout_1(batch_sequences))

        batch_sequences = batch_sequences + self._out_mlp(batch_sequences)

        batch_sequences = self._norm_2(self._dropout_2(batch_sequences))

        mask = pwF.create_mask_from_length(batch_sequence_lengths, batch_sequences.shape[1],
                                           self._is_end_padded).unsqueeze(-1)
        batch_sequences = batch_sequences.masked_fill(mask == 0, 0)

        return batch_sequences
