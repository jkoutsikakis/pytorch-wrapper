import torch
import unittest
import numpy as np

from pytorch_wrapper import modules


class EmbeddingLayerTestCase(unittest.TestCase):

    def test_randomly_initialized_non_trainable(self):

        vocab_size = 10
        emb_size = 20
        trainable = False
        padding_idx = 0

        layer = modules.EmbeddingLayer(vocab_size, emb_size, trainable, padding_idx)

        x = torch.tensor(
            [
                [1, 2, 0],
                [1, 2, 3]
            ], dtype=torch.long)

        x = layer(x)

        self.assertListEqual(list(x.shape), [2, 3, 20])
        self.assertListEqual(x[0][2].tolist(), torch.tensor([0] * emb_size).tolist())

    def test_randomly_initialized_trainable(self):

        vocab_size = 10
        emb_size = 20
        trainable = True
        padding_idx = 0

        layer = modules.EmbeddingLayer(vocab_size, emb_size, trainable, padding_idx)

        x = torch.tensor(
            [
                [1, 2, 0],
                [1, 2, 3]
            ], dtype=torch.long)

        x = layer(x)
        x.sum().backward()

        grad = next(layer._embedding_layer.parameters()).grad

        self.assertListEqual(list(x.shape), [2, 3, 20])
        self.assertListEqual(grad[0].tolist(), torch.tensor([0] * emb_size).tolist())

    def test_non_randomly_initialized_non_trainable(self):

        vocab_size = 3
        emb_size = 6
        trainable = False
        padding_idx = 0

        embeddings = np.array([
            [1, 2, 3, 4, 5, 6],
            [2, 5, 1, 2, 3, 4],
            [9, 1, 7, 2, 3, 7]
        ])

        layer = modules.EmbeddingLayer(vocab_size, emb_size, trainable, padding_idx)
        layer.load_embeddings(embeddings)

        x = torch.tensor(
            [
                [1, 2, 0],
                [1, 2, 1]
            ], dtype=torch.long)

        x = layer(x)

        self.assertListEqual(list(x.shape), [2, 3, emb_size])
        self.assertListEqual(x[0][2].tolist(), torch.tensor(embeddings[0]).tolist())

    def test_non_randomly_initialized_trainable(self):

        vocab_size = 3
        emb_size = 6
        trainable = True
        padding_idx = 0

        embeddings = np.array([
            [1, 2, 3, 4, 5, 6],
            [2, 5, 1, 2, 3, 4],
            [9, 1, 7, 2, 3, 7]
        ])

        layer = modules.EmbeddingLayer(vocab_size, emb_size, trainable, padding_idx)
        layer.load_embeddings(embeddings)

        x = torch.tensor(
            [
                [1, 2, 0],
                [1, 2, 1]
            ], dtype=torch.long)

        x = layer(x)
        x.sum().backward()

        grad = next(layer._embedding_layer.parameters()).grad

        self.assertListEqual(list(x.shape), [2, 3, emb_size])
        self.assertListEqual(grad[0].tolist(), torch.tensor([0] * emb_size).tolist())
