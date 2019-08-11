import unittest

from pytorch_wrapper import samplers


class SubsetSequentialSampler(unittest.TestCase):

    def test_len(self):
        indices = [1, 10, 20]
        sampler = samplers.SubsetSequentialSampler(indices)
        self.assertEqual(len(sampler), 3)

    def test_iter(self):
        indices = [1, 10, 20]
        out = []
        sampler = samplers.SubsetSequentialSampler(indices)
        for idx in sampler:
            out.append(idx)
        self.assertListEqual(indices, out)
