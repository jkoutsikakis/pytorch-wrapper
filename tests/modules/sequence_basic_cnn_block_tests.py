import torch
import unittest
import torch.nn as nn

from pytorch_wrapper import modules


class SequenceBasicCNNBlockTestCase(unittest.TestCase):

    def test_execution(self):
        batch_sequences = torch.tensor([
            [[1, 2, 3],
             [4, 5, 6],
             [1, 2, 3]],
            [[1, 2, 3],
             [6, 4, 3],
             [1, 2, 3]]], dtype=torch.float32
        )

        model = modules.SequenceBasicCNNBlock(batch_sequences.shape[-1],
                                              kernel_height=3,
                                              out_channels=10,
                                              activation=nn.ReLU,
                                              dp=0.5)

        res = model(batch_sequences)
        res.sum().backward()

        correct_shape = list(batch_sequences.shape)
        correct_shape[-1] = 10
        res_shape = list(res.shape)

        self.assertListEqual(res_shape, correct_shape)

