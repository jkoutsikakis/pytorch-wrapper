import torch
import unittest

from pytorch_wrapper import modules


class MultiHeadAttentionTestCase(unittest.TestCase):

    def test_dot_execution(self):
        q = torch.tensor(
            [[[1, 2, 3, 4],
              [4, 5, 6, 4],
              [4, 5, 6, 4]],
             [[1, 2, 3, 4],
              [6, 4, 3, 4],
              [4, 5, 6, 4]]],
            dtype=torch.float32
        )
        q_l = torch.tensor([1, 3])

        k = torch.tensor(
            [[[1, 2, 3, 4],
              [4, 5, 6, 4],
              [4, 5, 6, 4],
              [1, 2, 3, 4]],
             [[1, 2, 3, 4],
              [6, 4, 3, 4],
              [4, 5, 6, 4],
              [1, 2, 3, 4]]],
            dtype=torch.float32
        )
        k_l = torch.tensor([2, 4])

        v = torch.tensor(
            [[[1, 2, 3, 4, 3, 4],
              [4, 5, 6, 4, 3, 4],
              [4, 5, 6, 4, 3, 4],
              [1, 2, 3, 4, 3, 4]],
             [[1, 2, 3, 4, 3, 4],
              [6, 4, 3, 4, 3, 4],
              [4, 5, 6, 4, 3, 4],
              [1, 2, 3, 4, 3, 4]]],
            dtype=torch.float32
        )

        model = modules.MultiHeadAttention(
            q.shape[-1],
            k.shape[-1],
            v.shape[-1],
            heads=2,
            dp=0.1,
            attention_type='dot',
            is_end_padded=True
        )

        res = model(q, k, v, q_l, k_l)['output']

        self.assertListEqual(list(res.shape), [2, 3, 6])

    def test_multiplicative_execution(self):
        q = torch.tensor(
            [[[1, 2, 3, 4, 3, 4, 5, 6],
              [4, 5, 6, 4, 3, 4, 5, 6],
              [4, 5, 6, 4, 3, 4, 5, 6]],
             [[1, 2, 3, 4, 3, 4, 5, 6],
              [6, 4, 3, 4, 3, 4, 5, 6],
              [4, 5, 6, 4, 3, 4, 5, 6]]],
            dtype=torch.float32
        )
        q_l = torch.tensor([1, 3])

        k = torch.tensor(
            [[[1, 2, 3, 4],
              [4, 5, 6, 4],
              [4, 5, 6, 4],
              [1, 2, 3, 4]],
             [[1, 2, 3, 4],
              [6, 4, 3, 4],
              [4, 5, 6, 4],
              [1, 2, 3, 4]]],
            dtype=torch.float32
        )
        k_l = torch.tensor([2, 4])

        v = torch.tensor(
            [[[1, 2, 3, 4, 3, 4],
              [4, 5, 6, 4, 3, 4],
              [4, 5, 6, 4, 3, 4],
              [1, 2, 3, 4, 3, 4]],
             [[1, 2, 3, 4, 3, 4],
              [6, 4, 3, 4, 3, 4],
              [4, 5, 6, 4, 3, 4],
              [1, 2, 3, 4, 3, 4]]],
            dtype=torch.float32
        )

        model = modules.MultiHeadAttention(
            q.shape[-1],
            k.shape[-1],
            v.shape[-1],
            heads=2,
            dp=0.1,
            attention_type='multiplicative',
            is_end_padded=True
        )

        res = model(q, k, v, q_l, k_l)['output']
        res.sum().backward()

        self.assertListEqual(list(res.shape), [2, 3, 6])

    def test_additive_execution(self):
        q = torch.tensor(
            [[[1, 2, 3, 4, 3, 4, 5, 6],
              [4, 5, 6, 4, 3, 4, 5, 6],
              [4, 5, 6, 4, 3, 4, 5, 6]],
             [[1, 2, 3, 4, 3, 4, 5, 6],
              [6, 4, 3, 4, 3, 4, 5, 6],
              [4, 5, 6, 4, 3, 4, 5, 6]]],
            dtype=torch.float32
        )
        q_l = torch.tensor([1, 3])

        k = torch.tensor(
            [[[1, 2, 3, 4],
              [4, 5, 6, 4],
              [4, 5, 6, 4],
              [1, 2, 3, 4]],
             [[1, 2, 3, 4],
              [6, 4, 3, 4],
              [4, 5, 6, 4],
              [1, 2, 3, 4]]],
            dtype=torch.float32
        )
        k_l = torch.tensor([2, 4])

        v = torch.tensor(
            [[[1, 2, 3, 4, 3, 4],
              [4, 5, 6, 4, 3, 4],
              [4, 5, 6, 4, 3, 4],
              [1, 2, 3, 4, 3, 4]],
             [[1, 2, 3, 4, 3, 4],
              [6, 4, 3, 4, 3, 4],
              [4, 5, 6, 4, 3, 4],
              [1, 2, 3, 4, 3, 4]]],
            dtype=torch.float32
        )

        model = modules.MultiHeadAttention(
            q.shape[-1],
            k.shape[-1],
            v.shape[-1],
            heads=2,
            dp=0.1,
            attention_type='additive',
            is_end_padded=True
        )

        res = model(q, k, v, q_l, k_l)['output']
        res.sum().backward()

        self.assertListEqual(list(res.shape), [2, 3, 6])
