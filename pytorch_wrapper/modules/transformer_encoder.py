import torch.nn as nn

from . import TransformerEncoderBlock, SinusoidalPositionalEmbeddingLayer, MLP


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder (https://arxiv.org/pdf/1706.03762.pdf).
    """

    def __init__(self, time_step_size, heads, depth, dp=0, use_positional_embeddings=True, is_end_padded=True):
        """
        :param time_step_size: Time step size.
        :param heads: Number of attention heads.
        :param depth: Number of transformer blocks.
        :param dp: Dropout probability.
        :param use_positional_embeddings: Whether to use positional embeddings.
        :param is_end_padded: Whether to mask at the end.
        """
        super(TransformerEncoder, self).__init__()

        self._module_list = list()

        if use_positional_embeddings:
            self._pos_emb = SinusoidalPositionalEmbeddingLayer(time_step_size, pad_at_end=is_end_padded)
        else:
            self._pos_emb = None

        for _ in range(depth):
            cur_mlp = MLP(time_step_size,
                          hidden_layer_size=time_step_size,
                          hidden_activation=nn.ReLU,
                          output_size=time_step_size)
            cur_encoder_block = TransformerEncoderBlock(time_step_size, heads, cur_mlp, dp, is_end_padded)
            self._module_list.append(cur_encoder_block)

        self._module_list = nn.ModuleList(self._module_list)

    def forward(self, batch_sequences, batch_sequence_lengths):
        """
        :param batch_sequences: batch_sequences: 3D Tensor (batch_size, sequence_length, time_step_size).
        :param batch_sequence_lengths: 1D Tensor (batch_size) containing the lengths of the sequences.
        :return: 3D Tensor (batch_size, sequence_length, time_step_size).
        """

        if self._pos_emb is not None:
            batch_sequences = self._pos_emb(batch_sequence_lengths, batch_sequences.shape[1]) + batch_sequences

        for module in self._module_list:
            batch_sequences = module(batch_sequences, batch_sequence_lengths)

        return batch_sequences
