import torch
import unittest

from pytorch_wrapper import modules


class SinusoidalPositionalEmbeddingsLayerTestCase(unittest.TestCase):

    def test_execution(self):
        emb_size = 20
        pad_at_end = True
        init_max_sentence_length = 5

        layer = modules.SinusoidalPositionalEmbeddingLayer(emb_size, pad_at_end, init_max_sentence_length)

        length_tensor = torch.tensor([1, 2, 2], dtype=torch.long)
        max_sequence_length = 2

        res = layer(length_tensor, max_sequence_length)

        self.assertListEqual(list(res.shape), [3, 2, 20])
        self.assertListEqual(res[0][1].tolist(), torch.tensor([0] * emb_size).tolist())

    def test_small_init_max_size_execution(self):
        emb_size = 20
        pad_at_end = True
        init_max_sentence_length = 4

        layer = modules.SinusoidalPositionalEmbeddingLayer(emb_size, pad_at_end, init_max_sentence_length)

        length_tensor = torch.tensor([1, 2, 5], dtype=torch.long)
        max_sequence_length = 5

        res = layer(length_tensor, max_sequence_length)

        self.assertListEqual(list(res.shape), [3, 5, 20])
        self.assertListEqual(res[0][1].tolist(), torch.tensor([0] * emb_size).tolist())
