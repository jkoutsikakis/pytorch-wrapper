import torch
import unittest

from pytorch_wrapper import modules


class TransformerEncoderTestCase(unittest.TestCase):

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

        model = modules.TransformerEncoder(batch_sequences.shape[-1],
                                           heads=2,
                                           depth=4,
                                           dp=0.5,
                                           use_positional_embeddings=True,
                                           is_end_padded=True)

        res = model(batch_sequences, batch_sequence_lengths)
        res.sum().backward()

        correct_shape = list(batch_sequences.shape)
        res_shape = list(res.shape)

        self.assertListEqual(res_shape, correct_shape)
