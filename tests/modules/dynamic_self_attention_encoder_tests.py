import torch
import unittest
import torch.nn as nn

from pytorch_wrapper import modules


class DynamicSelfAttentionEncoderTestCase(unittest.TestCase):

    def test_execution(self):
        batch_sequences = torch.tensor([
            [[1, 2, 3],
             [4, 5, 6],
             [1, 2, 3]],
            [[1, 2, 3],
             [6, 4, 3],
             [1, 2, 3]]], dtype=torch.float32
        )
        batch_sequence_lengths = torch.tensor([1, 3])

        model = modules.DynamicSelfAttentionEncoder(batch_sequences.shape[-1],
                                                    att_scores_nb=1,
                                                    att_iterations=2,
                                                    projection_size=10,
                                                    projection_activation=nn.LeakyReLU,
                                                    attended_representation_activation=nn.Tanh,
                                                    is_end_padded=True)

        res = model(batch_sequences, batch_sequence_lengths)
        res.sum().backward()

        correct_shape = [batch_sequences.shape[0], 10]
        res_shape = list(res.shape)

        self.assertListEqual(res_shape, correct_shape)
