import torch.nn as nn

from .. import functional as pwF


class SequenceBasicCNNBlock(nn.Module):
    """
    Sequence Basic CNN Block.
    """

    def __init__(self,
                 time_step_size,
                 kernel_height=3,
                 out_channels=300,
                 activation=nn.ReLU,
                 dp=0):
        """
        :param time_step_size: Time step size.
        :param kernel_height: Filter height.
        :param out_channels: Number of filters.
        :param activation: Callable that creates the activation function.
        :param dp: Dropout probability.
        """

        super(SequenceBasicCNNBlock, self).__init__()

        self._kernel_height = kernel_height

        self._conv_layer = nn.Conv1d(in_channels=time_step_size, out_channels=out_channels, kernel_size=kernel_height)

        if activation is not None:
            self._activation = activation()
        else:
            self._activation = None

        self._dp = nn.Dropout(dp) if dp > 0 else None

    def forward(self, batch_sequences):
        """
        :param batch_sequences: 3D Tensor (batch_size, sequence_length, time_step_size) containing the sequence.
        :return: 2D Tensor (batch_size, sequence_length, out_channels) containing the encodings.
        """

        batch_sequences = pwF.pad(batch_sequences, self._kernel_height - 1, dim=1, pad_at_end=False)

        batch_sequences = self._conv_layer(batch_sequences.transpose(1, 2)).transpose(1, 2)

        if self._activation is not None:
            batch_sequences = self._activation(batch_sequences)

        if self._dp is not None:
            batch_sequences = self._dp(batch_sequences)

        return batch_sequences

