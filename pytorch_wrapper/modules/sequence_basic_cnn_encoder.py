import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import functional as pwF


class SequenceBasicCNNEncoder(nn.Module):
    """
    Basic CNN Encoder for sequences (https://arxiv.org/abs/1408.5882).
    """

    def __init__(self,
                 time_step_size,
                 input_activation=None,
                 kernel_heights=(1, 2, 3, 4, 5),
                 out_channels=300,
                 pre_pooling_activation=nn.ReLU,
                 pooling_function=F.max_pool1d,
                 post_pooling_activation=None,
                 post_pooling_dp=0):
        """
        :param time_step_size: Time step size.
        :param input_activation: Callable that creates the activation used on the input.
        :param kernel_heights: Tuple containing filter heights.
        :param out_channels: Number of filters for each filter height.
        :param pre_pooling_activation: Callable that creates the activation used before pooling.
        :param pooling_function: Callable that performs a pooling function before the activation.
        :param post_pooling_activation: Callable that creates the activation used after pooling.
        :param post_pooling_dp: Callable that performs a pooling function before the activation.
        """

        super(SequenceBasicCNNEncoder, self).__init__()

        self._min_len = max(kernel_heights)
        self._kernel_heights = kernel_heights

        self._input_activation = input_activation() if input_activation is not None else input_activation

        self._convolutional_layers = nn.ModuleList(
            modules=[nn.Conv1d(in_channels=time_step_size, out_channels=out_channels, kernel_size=kernel_height)
                     for kernel_height in kernel_heights]
        )

        if pre_pooling_activation is not None:
            self._pre_pooling_activation = pre_pooling_activation()
        else:
            self._pre_pooling_activation = None

        self._pooling_function = pooling_function

        if post_pooling_activation is not None:
            self._post_pooling_activation = post_pooling_activation()
        else:
            self._post_pooling_activation = None

        self._output_dp_layer = nn.Dropout(post_pooling_dp) if post_pooling_dp > 0 else None

    def forward(self, batch_sequences):
        """
        :param batch_sequences: 3D Tensor (batch_size, sequence_length, time_step_size) containing the sequence.
        :return: 2D Tensor (batch_size, len(kernel_heights) * out_channels) containing the encodings.
        """

        if self._min_len > batch_sequences.shape[1]:
            batch_sequences = pwF.pad(batch_sequences, self._min_len - batch_sequences.shape[1], dim=1, pad_at_end=False)

        convolutions = [conv(batch_sequences.transpose(1, 2)) for conv in self._convolutional_layers]

        if self._pre_pooling_activation is not None:
            convolutions = [self._pre_pooling_activation(c) for c in convolutions]

        pooled = [self._pooling_function(c, c.shape[2]).squeeze(2) for c in convolutions]

        if self._post_pooling_activation is not None:
            pooled = [self._post_pooling_activation(p) for p in pooled]

        if len(self._kernel_heights) > 1:
            output = torch.cat(pooled, dim=1)
        else:
            output = pooled[0]

        if self._output_dp_layer is not None:
            output = self._output_dp_layer(output)

        return output
