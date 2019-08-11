import torch
import unittest

from unittest.mock import MagicMock

from pytorch_wrapper import modules


class TransformerEncoderBlockTestCase(unittest.TestCase):

    def test_execution(self):
        batch_sequences = torch.tensor([
            [[1, 2, 3, 6],
             [4, 5, 6, 8],
             [1, 2, 3, 9]],
            [[1, 2, 3, 2],
             [6, 4, 3, 1],
             [1, 2, 3, 5]]], dtype=torch.float32
        )
        batch_sequence_lengths = torch.tensor([1, 3])

        mocked_out_mlp = MagicMock(return_value=0)

        model = modules.TransformerEncoderBlock(batch_sequences.shape[-1],
                                                heads=2,
                                                out_mlp=mocked_out_mlp,
                                                dp=0.5,
                                                is_end_padded=True)

        res = model(batch_sequences, batch_sequence_lengths)
        res.sum().backward()

        correct_shape = list(batch_sequences.shape)
        res_shape = list(res.shape)

        self.assertListEqual(res_shape, correct_shape)
