import torch
import unittest
import torch.nn as nn

from pytorch_wrapper import modules


class SequenceDenseCNNTestCase(unittest.TestCase):

    def test_execution(self):
        batch_sequences = torch.tensor([
            [[1, 2, 3],
             [4, 5, 6],
             [1, 2, 3]],
            [[1, 2, 3],
             [6, 4, 3],
             [1, 2, 3]]], dtype=torch.float32
        )

        model = modules.SequenceDenseCNN(batch_sequences.shape[-1],
                                         projection_layer_size=5,
                                         kernel_heights=(3, 5),
                                         feature_map_increase=5,
                                         cnn_depth=3,
                                         output_projection_layer_size=10,
                                         activation=nn.LeakyReLU,
                                         dp=0.5,
                                         normalize_output=True)

        res = model(batch_sequences)
        res.sum().backward()

        correct_shape = list(batch_sequences.shape)
        correct_shape[-1] = 10
        res_shape = list(res.shape)

        self.assertListEqual(res_shape, correct_shape)
