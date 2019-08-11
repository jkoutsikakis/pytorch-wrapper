import torch.nn as nn

from .layer_norm import LayerNorm


class MLP(nn.Module):
    """
    Multi Layer Perceptron.
    """

    def __init__(self,

                 input_size,
                 input_activation=None,
                 input_dp=None,
                 input_pre_activation_bn=False,
                 input_post_activation_bn=False,
                 input_pre_activation_ln=False,
                 input_post_activation_ln=False,

                 num_hidden_layers=1,
                 hidden_layer_size=128,
                 hidden_layer_bias=True,
                 hidden_layer_init=lambda weights: nn.init.xavier_uniform_(weights),
                 hidden_layer_bias_init=lambda bias: nn.init.constant_(bias, 0),
                 hidden_activation=nn.ReLU,
                 hidden_dp=None,
                 hidden_layer_pre_activation_bn=False,
                 hidden_layer_post_activation_bn=False,
                 hidden_layer_pre_activation_ln=False,
                 hidden_layer_post_activation_ln=False,

                 output_layer_init=lambda weights: nn.init.xavier_uniform_(weights),
                 output_layer_bias_init=lambda bias: nn.init.constant_(bias, 0),
                 output_size=1,
                 output_layer_bias=True,
                 output_activation=None,
                 output_dp=None,
                 output_layer_pre_activation_bn=False,
                 output_layer_post_activation_bn=False,
                 output_layer_pre_activation_ln=False,
                 output_layer_post_activation_ln=False):

        """
        :param input_size: Size of the last dimension of the input.
        :param input_activation: Callable that creates the activation used on the input.
        :param input_dp: Callable that creates the activation used on the input.
        :param input_pre_activation_bn: Whether to use batch normalization before the activation of the input
            layer.
        :param input_post_activation_bn: Whether to use batch normalization after the activation of the input
            layer.
        :param input_pre_activation_ln: Whether to use layer normalization before the activation of the input
            layer.
        :param input_post_activation_ln: Whether to use layer normalization after the activation of the input
            layer.
        :param num_hidden_layers: Number of hidden layers.
        :param hidden_layer_size: Size of hidden layers. It is also possible to provide a list containing a different
            size for each hidden layer.
        :param hidden_layer_bias: Whether to use bias. It is also possible to provide a list containing a different
            option for each hidden layer.
        :param hidden_layer_init: Callable that initializes inplace the weights of the hidden layers.
        :param hidden_layer_bias_init: Callable that initializes inplace the bias of the hidden layers.
        :param hidden_activation: Callable that creates the activation used after each hidden layer. It is also possible
            to provide a list containing num_hidden_layers callables.
        :param hidden_dp: Dropout probability for the hidden layers. It is also possible to provide a list containing
            num_hidden_layers probabilities.
        :param hidden_layer_pre_activation_bn: Whether to use batch normalization before the activation of each hidden
            layer.
        :param hidden_layer_post_activation_bn: Whether to use batch normalization after the activation of each hidden
            layer.
        :param hidden_layer_pre_activation_ln: Whether to use layer normalization before the activation of each hidden
            layer.
        :param hidden_layer_post_activation_ln: Whether to use layer normalization after the activation of each hidden
            layer.
        :param output_layer_init: Callable that initializes inplace the weights of the output layer.
        :param output_layer_bias_init: Callable that initializes inplace the bias of the output layer.
        :param output_size: Output size.
        :param output_layer_bias: Whether to use bias.
        :param output_activation: Callable that creates the activation used after the output layer.
        :param output_dp: Dropout probability for the output layer.
        :param output_layer_pre_activation_bn: Whether to use batch normalization before the activation of the output
            layer.
        :param output_layer_post_activation_bn: Whether to use batch normalization before the activation of the output
            layer.
        :param output_layer_pre_activation_ln: Whether to use layer normalization before the activation of the output
            layer.
        :param output_layer_post_activation_ln: Whether to use layer normalization before the activation of the output
            layer.
        """

        super(MLP, self).__init__()

        if type(hidden_layer_size) == list and len(hidden_layer_size) != num_hidden_layers or type(
                hidden_activation) == list and len(hidden_activation) != num_hidden_layers or type(
                hidden_dp) == list and len(hidden_dp) != num_hidden_layers:
            raise ValueError("Wrong parameters")

        if type(hidden_layer_size) != list:
            hidden_layer_size = [hidden_layer_size] * num_hidden_layers

        if type(hidden_layer_bias) != list:
            hidden_layer_bias = [hidden_layer_bias] * num_hidden_layers

        if type(hidden_activation) != list:
            if hidden_activation is not None:
                hidden_activation = [hidden_activation() for _ in range(num_hidden_layers)]
            else:
                hidden_activation = [None] * num_hidden_layers

        if type(hidden_dp) != list:
            hidden_dp = [hidden_dp] * num_hidden_layers

        self._input_pre_activation_bn = nn.BatchNorm1d(input_size) if input_pre_activation_bn else None
        self._input_pre_activation_ln = LayerNorm(input_size) if input_pre_activation_ln else None

        self._input_activation = input_activation() if input_activation is not None else None

        if input_dp is not None and input_dp > 0:
            self._input_dp_layer = nn.Dropout(input_dp)
        else:
            self._input_dp_layer = None

        self._input_post_activation_bn = nn.BatchNorm1d(input_size) if input_post_activation_bn else None
        self._input_post_activation_ln = LayerNorm(input_size) if input_post_activation_ln else None

        self._hidden_layer_list = None

        if num_hidden_layers > 0:
            self._hidden_layer_list = nn.ModuleList()

            cur_linear = nn.Linear(input_size, hidden_layer_size[0], hidden_layer_bias[0])
            if hidden_layer_init is not None:
                hidden_layer_init(cur_linear.weight)
            if hidden_layer_bias_init is not None and hidden_layer_bias[0]:
                hidden_layer_bias_init(cur_linear.bias)

            self._hidden_layer_list.append(cur_linear)

            if hidden_layer_pre_activation_bn:
                self._hidden_layer_list.append(nn.BatchNorm1d(hidden_layer_size[0]))

            if hidden_layer_pre_activation_ln:
                self._hidden_layer_list.append(LayerNorm(hidden_layer_size[0]))

            if hidden_activation[0] is not None:
                self._hidden_layer_list.append(hidden_activation[0])

            if hidden_layer_post_activation_bn:
                self._hidden_layer_list.append(nn.BatchNorm1d(hidden_layer_size[0]))

            if hidden_layer_post_activation_ln:
                self._hidden_layer_list.append(LayerNorm(hidden_layer_size[0]))

            if hidden_dp[0] is not None and hidden_dp[0] > 0:
                self._hidden_layer_list.append(nn.Dropout(hidden_dp[0]))

            for i in range(1, num_hidden_layers):

                cur_linear = nn.Linear(hidden_layer_size[i - 1], hidden_layer_size[i], hidden_layer_bias[i])
                if hidden_layer_init is not None:
                    hidden_layer_init(cur_linear.weight)
                if hidden_layer_bias_init is not None and hidden_layer_bias[i]:
                    hidden_layer_bias_init(cur_linear.bias)

                self._hidden_layer_list.append(cur_linear)

                if hidden_layer_pre_activation_bn:
                    self._hidden_layer_list.append(nn.BatchNorm1d(hidden_layer_size[i]))

                if hidden_layer_pre_activation_ln:
                    self._hidden_layer_list.append(LayerNorm(hidden_layer_size[i]))

                if hidden_activation[i] is not None:
                    self._hidden_layer_list.append(hidden_activation[i])

                if hidden_dp[i] is not None and hidden_dp[i] > 0:
                    self._hidden_layer_list.append(nn.Dropout(hidden_dp[i]))

                if hidden_layer_post_activation_bn:
                    self._hidden_layer_list.append(nn.BatchNorm1d(hidden_layer_size[i]))

                if hidden_layer_post_activation_ln:
                    self._hidden_layer_list.append(LayerNorm(hidden_layer_size[i]))

            self._output_layer = nn.Linear(hidden_layer_size[-1], output_size, output_layer_bias)

        else:
            self._output_layer = nn.Linear(input_size, output_size, output_layer_bias)

        if output_layer_init is not None:
            output_layer_init(self._output_layer.weight)
        if output_layer_bias_init is not None and output_layer_bias:
            output_layer_bias_init(self._output_layer.bias)

        self._output_layer_pre_activation_bn = nn.BatchNorm1d(output_size) if output_layer_pre_activation_bn else None
        self._output_layer_pre_activation_ln = LayerNorm(output_size) if output_layer_pre_activation_ln else None

        self._output_activation = output_activation() if output_activation is not None else None

        self._output_layer_post_activation_bn = nn.BatchNorm1d(output_size) if output_layer_post_activation_bn else None
        self._output_layer_post_activation_ln = LayerNorm(output_size) if output_layer_post_activation_ln else None

        if output_dp is not None and output_dp > 0:
            self._output_dp_layer = nn.Dropout(output_dp)
        else:
            self._output_dp_layer = None

    def forward(self, x):
        """
        :param x: Tensor having its last dimension being of size input_size.
        :return: Tensor with the same shape as x except the last dimension which is of size output_size.
        """

        if self._input_pre_activation_bn is not None:
            x = self._input_pre_activation_bn(x)

        if self._input_pre_activation_ln is not None:
            x = self._input_pre_activation_ln(x)

        if self._input_activation is not None:
            x = self._input_activation(x)

        if self._input_dp_layer is not None:
            x = self._input_dp_layer(x)

        if self._input_post_activation_bn is not None:
            x = self._input_post_activation_bn(x)

        if self._input_post_activation_ln is not None:
            x = self._input_post_activation_ln(x)

        if self._hidden_layer_list is not None:
            for hidden_layer in self._hidden_layer_list:
                x = hidden_layer(x)

        x = self._output_layer(x)

        if self._output_layer_pre_activation_bn is not None:
            x = self._output_layer_pre_activation_bn(x)

        if self._output_layer_pre_activation_ln is not None:
            x = self._output_layer_pre_activation_ln(x)

        if self._output_activation is not None:
            x = self._output_activation(x)

        if self._output_dp_layer is not None:
            x = self._output_dp_layer(x)

        if self._output_layer_post_activation_bn is not None:
            x = self._output_layer_post_activation_bn(x)

        if self._output_layer_post_activation_ln is not None:
            x = self._output_layer_post_activation_ln(x)

        return x
