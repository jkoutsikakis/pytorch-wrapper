import torch
import unittest

from pytorch_wrapper import modules


class LayerNormTestCase(unittest.TestCase):

    def test_execution(self):
        batch_sequences = torch.tensor([
            [[1, 2, 3],
             [4, 5, 6],
             [1, 2, 3]],
            [[1, 2, 3],
             [6, 4, 3],
             [1, 2, 3]]], dtype=torch.float32
        )

        model = modules.LayerNorm(batch_sequences.shape[-1])
        res = model(batch_sequences)
        res.sum().backward()

        correct_shape = list(batch_sequences.shape)
        res_shape = list(res.shape)

        self.assertListEqual(res_shape, correct_shape)
