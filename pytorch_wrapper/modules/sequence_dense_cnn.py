import torch
import torch.nn as nn

from .. import functional as pwF


class SequenceDenseCNN(nn.Module):
    """
    Dense CNN for sequences (https://arxiv.org/abs/1808.07383).
    """

    def __init__(self,
                 input_size,
                 projection_layer_size=150,
                 kernel_heights=(3, 5),
                 feature_map_increase=75,
                 cnn_depth=3,
                 output_projection_layer_size=300,
                 activation=nn.LeakyReLU,
                 dp=0,
                 normalize_output=True):
        """
        :param input_size: Time step size.
        :param projection_layer_size: Size of projection_layer.
        :param kernel_heights: Kernel height of the filters.
        :param feature_map_increase: Number of filters of each convolutional layer.
        :param cnn_depth: Number of convolutional layers per kernel height.
        :param output_projection_layer_size: Size of the output time_steps.
        :param activation: Callable that creates the activation used after each layer.
        :param dp: Dropout probability.
        :param normalize_output: Whether to perform l2 normalization on the output.
        """

        super(SequenceDenseCNN, self).__init__()

        self._kernel_heights = kernel_heights
        self._normalize_output = normalize_output

        self._convolutional_blocks = nn.ModuleList()

        for kernel_height in kernel_heights:
            current_convolutional_block = nn.ModuleList()

            input_projection_layer = list()

            input_projection_layer.append(
                nn.Conv1d(in_channels=input_size, out_channels=projection_layer_size, kernel_size=1))
            input_projection_layer.append(activation())
            if dp > 0:
                input_projection_layer.append(nn.Dropout(dp))

            input_projection_layer = nn.Sequential(*input_projection_layer)
            current_convolutional_block.append(input_projection_layer)

            for layer_depth in range(cnn_depth):
                current_layer = list()
                current_layer.append(nn.Conv1d(in_channels=projection_layer_size + layer_depth * feature_map_increase,
                                               out_channels=feature_map_increase,
                                               kernel_size=kernel_height))
                current_layer.append(activation())
                if dp > 0:
                    current_layer.append(nn.Dropout(dp))

                current_convolutional_block.append(nn.Sequential(*current_layer))

            self._convolutional_blocks.append(current_convolutional_block)

        self._output_projection_layer = list()

        conv_result_size = input_size + (projection_layer_size + cnn_depth * feature_map_increase) * len(kernel_heights)
        self._output_projection_layer.append(
            nn.Conv1d(in_channels=conv_result_size, out_channels=output_projection_layer_size, kernel_size=1))
        self._output_projection_layer.append(activation())
        if dp > 0:
            self._output_projection_layer.append(nn.Dropout(dp))

        self._output_projection_layer = nn.Sequential(*self._output_projection_layer)

    def forward(self, batch_sequences):
        """
        :param batch_sequences: 3D Tensor (batch_size, sequence_length, time_step_size).
        :return: 3D Tensor (batch_size, sequence_length, output_projection_layer_size).
        """

        batch_sequences = batch_sequences.transpose(1, 2)
        output = [batch_sequences]

        for i, convolutional_block in enumerate(self._convolutional_blocks):

            residuals = [convolutional_block[0](batch_sequences)]

            for convolutional_layer_i in range(1, len(convolutional_block)):

                if len(residuals) == 1:
                    current_input = residuals[0]
                else:
                    current_input = torch.cat(residuals, dim=1)

                current_input = pwF.pad(current_input, self._kernel_heights[i] - 1, dim=2, pad_at_end=False)
                out = convolutional_block[convolutional_layer_i](current_input)

                residuals.append(out)

            output.extend(residuals)

        output_projected = self._output_projection_layer(torch.cat(output, dim=1))

        if self._normalize_output:
            output_projected_norm = torch.norm(output_projected, p=2, dim=1, keepdim=True).detach()
            output_projected = output_projected.div(output_projected_norm.expand_as(output_projected))

        output_projected = output_projected.transpose(1, 2)

        return output_projected
