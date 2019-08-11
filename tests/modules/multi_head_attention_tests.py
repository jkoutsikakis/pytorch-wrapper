import torch
import unittest

from pytorch_wrapper import modules


class MultiHeadAttentionTestCase(unittest.TestCase):

    def test_execution(self):
        batch_sequences = torch.tensor([
            [[1, 2, 3, 4],
             [4, 5, 6, 4],
             [1, 2, 3, 4]],
            [[1, 2, 3, 4],
             [6, 4, 3, 4],
             [1, 2, 3, 4]]], dtype=torch.float32
        )
        batch_sequence_lengths = torch.tensor([1, 3])

        model = modules.MultiHeadAttention(batch_sequences.shape[-1],
                                           heads=2,
                                           dp=0.1,
                                           is_end_padded=True)

        res = model(batch_sequences, batch_sequences, batch_sequences, batch_sequence_lengths, batch_sequence_lengths)
        res.sum().backward()

        self.assertListEqual(list(res.shape), list(batch_sequences.shape))
