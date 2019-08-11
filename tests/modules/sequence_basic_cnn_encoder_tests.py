import torch
import unittest
import torch.nn as nn
import torch.nn.functional as F

from pytorch_wrapper import modules


class SequenceBasicCNNEncoderTestCase(unittest.TestCase):

    def test_execution(self):
        batch_sequences = torch.tensor([
            [[1, 2, 3],
             [4, 5, 6],
             [1, 2, 3]],
            [[1, 2, 3],
             [6, 4, 3],
             [1, 2, 3]]], dtype=torch.float32
        )

        model = modules.SequenceBasicCNNEncoder(batch_sequences.shape[-1],
                                                input_activation=nn.ReLU,
                                                kernel_heights=(1, 3, 5),
                                                out_channels=5,
                                                pre_pooling_activation=nn.ReLU,
                                                pooling_function=F.max_pool1d,
                                                post_pooling_activation=nn.ReLU,
                                                post_pooling_dp=0.5)

        res = model(batch_sequences)
        res.sum().backward()

        correct_shape = [batch_sequences.shape[0], 3 * 5]
        res_shape = list(res.shape)

        self.assertListEqual(res_shape, correct_shape)
