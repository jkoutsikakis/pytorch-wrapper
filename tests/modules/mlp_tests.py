import torch
import unittest

from torch import nn

from pytorch_wrapper import modules


class MLPTestCase(unittest.TestCase):

    def test_execution(self):
        x = torch.tensor(
            [
                [1, 2, 3],
                [4, 5, 6],
                [1, 2, 3]
            ], dtype=torch.float32
        )

        model = modules.MLP(
            x.shape[-1],
            input_activation=nn.ReLU,
            input_dp=0.1,
            input_pre_activation_bn=True,
            input_post_activation_bn=True,
            input_pre_activation_ln=True,
            input_post_activation_ln=True,

            num_hidden_layers=1,
            hidden_layer_size=128,
            hidden_layer_bias=False,
            hidden_layer_init=lambda weights: nn.init.xavier_uniform_(weights),
            hidden_layer_bias_init=lambda bias: nn.init.constant_(bias, 0),
            hidden_activation=nn.ReLU,
            hidden_dp=0.1,
            hidden_layer_pre_activation_bn=True,
            hidden_layer_post_activation_bn=True,
            hidden_layer_pre_activation_ln=True,
            hidden_layer_post_activation_ln=True,

            output_layer_init=lambda weights: nn.init.xavier_uniform_(weights),
            output_layer_bias_init=lambda bias: nn.init.constant_(bias, 0),
            output_size=128,
            output_layer_bias=True,
            output_activation=nn.ReLU,
            output_dp=0.1,
            output_layer_pre_activation_bn=True,
            output_layer_post_activation_bn=True,
            output_layer_pre_activation_ln=True,
            output_layer_post_activation_ln=True)

        res = model(x)
        res.sum().backward()

        correct_shape = list(x.shape)
        correct_shape[-1] = 128
        res_shape = list(res.shape)

        self.assertListEqual(res_shape, correct_shape)
