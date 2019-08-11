import torch
import unittest

from unittest.mock import MagicMock

from pytorch_wrapper import modules


class SoftmaxSelfAttentionEncoderTestCase(unittest.TestCase):

    def test_single_attention_end_padded(self):
        batch_sequences = torch.tensor([
            [[1, 2, 3],
             [4, 5, 6],
             [1, 2, 3]],
            [[1, 2, 3],
             [6, 4, 3],
             [1, 2, 3]]], dtype=torch.float32
        )

        batch_sequence_lengths = torch.tensor([3, 2], dtype=torch.long)

        attention_scores = torch.tensor([
            [[5], [7], [9]],
            [[1], [2], [7]]], dtype=torch.float32, requires_grad=True)

        mocked_attention_mlp = MagicMock(return_value=attention_scores)
        is_end_padded = True

        model = modules.SoftmaxSelfAttentionEncoder(mocked_attention_mlp, is_end_padded)
        res = model(batch_sequences, batch_sequence_lengths)
        output = res['output']
        normalized_attention_scores = res['att_scores']
        output.sum().backward()

        correct_shape = [batch_sequences.shape[0], batch_sequences.shape[2]]
        output_shape = list(output.shape)

        self.assertListEqual(output_shape, correct_shape)
        self.assertAlmostEqual(normalized_attention_scores[1][2].item(), 0.)
        self.assertAlmostEqual(attention_scores.grad[1][2].item(), 0.)

    def test_single_attention_start_padded(self):
        batch_sequences = torch.tensor([
            [[1, 2, 3],
             [4, 5, 6],
             [1, 2, 3]],
            [[1, 2, 3],
             [6, 4, 3],
             [1, 2, 3]]], dtype=torch.float32
        )

        batch_sequence_lengths = torch.tensor([3, 2], dtype=torch.long)

        attention_scores = torch.tensor([
            [[5], [7], [9]],
            [[1], [2], [7]]], dtype=torch.float32, requires_grad=True)

        mocked_attention_mlp = MagicMock(return_value=attention_scores)
        is_end_padded = False

        model = modules.SoftmaxSelfAttentionEncoder(mocked_attention_mlp, is_end_padded)
        res = model(batch_sequences, batch_sequence_lengths)
        output = res['output']
        normalized_attention_scores = res['att_scores']
        output.sum().backward()

        correct_shape = [batch_sequences.shape[0], batch_sequences.shape[2]]
        output_shape = list(output.shape)

        self.assertListEqual(output_shape, correct_shape)
        self.assertAlmostEqual(normalized_attention_scores[1][0].item(), 0.)
        self.assertAlmostEqual(attention_scores.grad[1][0].item(), 0.)

    def test_multi_attention_end_padded(self):
        batch_sequences = torch.tensor([
            [[1, 2, 3],
             [4, 5, 6],
             [1, 2, 3]],
            [[1, 2, 3],
             [6, 4, 3],
             [1, 2, 3]]], dtype=torch.float32
        )

        batch_sequence_lengths = torch.tensor([3, 2], dtype=torch.long)

        attention_scores = torch.tensor([
            [[5, 2], [7, 3], [9, 4]],
            [[1, 5], [2, 6], [7, 8]]], dtype=torch.float32, requires_grad=True)

        mocked_attention_mlp = MagicMock(return_value=attention_scores)
        is_end_padded = True

        model = modules.SoftmaxSelfAttentionEncoder(mocked_attention_mlp, is_end_padded)
        res = model(batch_sequences, batch_sequence_lengths)
        output = res['output']
        normalized_attention_scores = res['att_scores']
        output.sum().backward()

        correct_shape = [batch_sequences.shape[0], 2, batch_sequences.shape[2]]
        output_shape = list(output.shape)

        self.assertListEqual(output_shape, correct_shape)
        self.assertAlmostEqual(normalized_attention_scores[1][2][0].item(), 0.)
        self.assertAlmostEqual(normalized_attention_scores[1][2][1].item(), 0.)
        self.assertAlmostEqual(attention_scores.grad[1][2][0].item(), 0.)
        self.assertAlmostEqual(attention_scores.grad[1][2][1].item(), 0.)

    def test_multi_attention_start_padded(self):
        batch_sequences = torch.tensor([
            [[1, 2, 3],
             [4, 5, 6],
             [1, 2, 3]],
            [[1, 2, 3],
             [6, 4, 3],
             [1, 2, 3]]], dtype=torch.float32
        )

        batch_sequence_lengths = torch.tensor([3, 2], dtype=torch.long)

        attention_scores = torch.tensor([
            [[5, 2], [7, 3], [9, 4]],
            [[1, 5], [2, 6], [7, 8]]], dtype=torch.float32, requires_grad=True)

        mocked_attention_mlp = MagicMock(return_value=attention_scores)
        is_end_padded = False

        model = modules.SoftmaxSelfAttentionEncoder(mocked_attention_mlp, is_end_padded)
        res = model(batch_sequences, batch_sequence_lengths)
        output = res['output']
        normalized_attention_scores = res['att_scores']
        output.sum().backward()

        correct_shape = [batch_sequences.shape[0], 2, batch_sequences.shape[2]]
        output_shape = list(output.shape)

        self.assertListEqual(output_shape, correct_shape)
        self.assertAlmostEqual(normalized_attention_scores[1][0][0].item(), 0.)
        self.assertAlmostEqual(normalized_attention_scores[1][0][1].item(), 0.)
        self.assertAlmostEqual(attention_scores.grad[1][0][0].item(), 0.)
        self.assertAlmostEqual(attention_scores.grad[1][0][1].item(), 0.)


