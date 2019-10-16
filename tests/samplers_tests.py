import unittest

from pytorch_wrapper import samplers


class SubsetSequentialSampler(unittest.TestCase):

    def test_len(self):
        indexes = [1, 10, 20]
        sampler = samplers.SubsetSequentialSampler(indexes)
        self.assertEqual(len(sampler), 3)

    def test_iter(self):
        indexes = [1, 10, 20]
        out = []
        sampler = samplers.SubsetSequentialSampler(indexes)
        for idx in sampler:
            out.append(idx)
        self.assertListEqual(indexes, out)


class OrderedBatchWiseRandomSampler(unittest.TestCase):

    def test_len(self):
        data_source = [1, 2, 1, 2, 1, 2]
        sampler = samplers.OrderedBatchWiseRandomSampler(data_source, lambda x: data_source[x], 3)
        self.assertEqual(len(sampler), 6)

    def test_iter(self):
        data_source = [1, 2, 1, 2, 1, 2]
        out = []
        sampler = samplers.OrderedBatchWiseRandomSampler(data_source, lambda x: data_source[x], 3)
        for idx in sampler:
            out.append(idx)

        for i in range(1, 3):
            self.assertEqual(data_source[out[i]], data_source[out[0]])
            self.assertEqual(data_source[out[i+3]], data_source[out[3]])


class SubsetOrderedBatchWiseRandomSampler(unittest.TestCase):

    def test_len(self):
        data_source = [0, 0, 1, 2, 1, 2, 1, 2]
        indexes = [2, 3, 4, 5, 6, 7]
        sampler = samplers.SubsetOrderedBatchWiseRandomSampler(indexes, lambda x: data_source[x], 3)
        self.assertEqual(len(sampler), 6)

    def test_iter(self):
        data_source = [0, 0, 1, 2, 1, 2, 1, 2]
        indexes = [2, 3, 4, 5, 6, 7]
        out = []
        sampler = samplers.SubsetOrderedBatchWiseRandomSampler(indexes, lambda x: data_source[x], 3)
        for idx in sampler:
            out.append(idx)
        self.assertTrue(
            (set(out[:3]) == {2, 4, 6} and set(out[3:]) == {3, 5, 7})
            or (set(out[3:]) == {2, 4, 6} and set(out[:3]) == {3, 5, 7})
        )


class OrderedSequentialSampler(unittest.TestCase):

    def test_len(self):
        data_source = [1, 10, 20]
        sampler = samplers.OrderedSequentialSampler(data_source, lambda x: -x)
        self.assertEqual(len(sampler), 3)

    def test_iter(self):
        data_source = [1, 10, 20]
        out = []
        sampler = samplers.OrderedSequentialSampler(data_source, lambda x: -x)
        for idx in sampler:
            out.append(idx)
        self.assertListEqual([2, 1, 0], out)


class SubsetOrderedSequentialSampler(unittest.TestCase):

    def test_len(self):
        indexes = [1, 10, 20]
        sampler = samplers.SubsetOrderedSequentialSampler(indexes, lambda x: -x)
        self.assertEqual(len(sampler), 3)

    def test_iter(self):
        indexes = [1, 10, 20]
        out = []
        sampler = samplers.SubsetOrderedSequentialSampler(indexes, lambda x: -x)
        for idx in sampler:
            out.append(idx)
        self.assertListEqual([20, 10, 1], out)
