import torch
import unittest
import numpy as np

from pytorch_wrapper import functional as pwF


class CreateMaskFromLengthTestCase(unittest.TestCase):

    def test_1D_end_padded(self):
        length_tensor = torch.tensor([1, 3])
        mask_size = 5
        is_end_padded = True

        result = pwF.create_mask_from_length(length_tensor, mask_size, is_end_padded)
        correct = torch.tensor(
            [
                [1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0]
            ],
            dtype=torch.uint8
        )

        self.assertListEqual(result.tolist(), correct.tolist())

    def test_1D_start_padded(self):
        length_tensor = torch.tensor([1, 3])
        mask_size = 5
        is_end_padded = False

        result = pwF.create_mask_from_length(length_tensor, mask_size, is_end_padded)
        correct = torch.tensor(
            [
                [0, 0, 0, 0, 1],
                [0, 0, 1, 1, 1]
            ],
            dtype=torch.uint8
        )

        self.assertListEqual(result.tolist(), correct.tolist())

    def test_2D_start_padded(self):
        length_tensor = torch.tensor([[1, 2], [3, 4]])
        mask_size = 5
        is_end_padded = False

        result = pwF.create_mask_from_length(length_tensor, mask_size, is_end_padded)
        correct = torch.tensor(
            [
                [
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 1]
                ],
                [
                    [0, 0, 1, 1, 1],
                    [0, 1, 1, 1, 1]
                ]
            ],
            dtype=torch.uint8
        )

        self.assertListEqual(result.tolist(), correct.tolist())


class MaskedMaxPoolingTestCase(unittest.TestCase):

    def test_3D(self):
        data_tensor = torch.tensor(
            [
                [
                    [8, 4, 0, 0, 1],
                    [0, 2, 6, 1, 3],
                    [0, 3, 7, 4, 7]
                ],
                [
                    [9, 7, 2, 7, 3],
                    [0, 3, 7, 4, 7],
                    [0, 3, 7, 4, 7]
                ]
            ],
            dtype=torch.float32
        )

        mask = torch.tensor(
            [
                [
                    [0, 0, 0, 0, 1],
                    [1, 1, 0, 1, 1],
                    [0, 0, 1, 1, 1]
                ],
                [
                    [1, 0, 1, 1, 1],
                    [0, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1]
                ]
            ],
            dtype=torch.uint8
        )

        dim = -2

        correct = torch.tensor(
            [
                [0, 2, 7, 4, 7],
                [9, 3, 7, 7, 7]
            ],
            dtype=torch.float32
        )

        result = pwF.masked_max_pooling(data_tensor, mask, dim)

        self.assertListEqual(result.tolist(), correct.tolist())

    def test_3D_broadcasting_mask(self):
        data_tensor = torch.tensor(
            [
                [
                    [8, 4, 0, 0, 1],
                    [0, 2, 6, 1, 3],
                    [0, 3, 7, 4, 7]
                ],
                [
                    [9, 7, 2, 7, 3],
                    [0, 3, 7, 4, 7],
                    [0, 3, 7, 4, 7]
                ]
            ],
            dtype=torch.float32
        )

        mask = torch.tensor(
            [
                [1, 1, 0],
                [1, 0, 0]
            ],
            dtype=torch.uint8
        )

        dim = -2

        correct = torch.tensor(
            [
                [8, 4, 6, 1, 3],
                [9, 7, 2, 7, 3]
            ],
            dtype=torch.float32
        )

        result = pwF.masked_max_pooling(data_tensor, mask, dim)

        self.assertListEqual(result.tolist(), correct.tolist())


class MaskedMinPoolingTestCase(unittest.TestCase):

    def test_3D(self):
        data_tensor = torch.tensor(
            [
                [
                    [8, 4, 0, 0, 1],
                    [0, 2, 6, 1, 3],
                    [0, 3, 7, 4, 7]
                ],
                [
                    [9, 7, 2, 7, 3],
                    [0, 3, 7, 4, 7],
                    [0, 3, 7, 4, 7]
                ]
            ],
            dtype=torch.float32
        )

        mask = torch.tensor(
            [
                [
                    [0, 0, 0, 0, 1],
                    [1, 1, 0, 1, 1],
                    [0, 0, 1, 1, 1]
                ],
                [
                    [1, 0, 1, 1, 1],
                    [0, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1]
                ]
            ],
            dtype=torch.uint8
        )

        dim = -2

        correct = torch.tensor(
            [
                [0, 2, 7, 1, 1],
                [9, 3, 2, 4, 3]
            ],
            dtype=torch.float32
        )

        result = pwF.masked_min_pooling(data_tensor, mask, dim)

        self.assertListEqual(result.tolist(), correct.tolist())

    def test_3D_broadcasting_mask(self):
        data_tensor = torch.tensor(
            [
                [
                    [8, 4, 0, 0, 1],
                    [0, 2, 6, 1, 3],
                    [0, 3, 7, 4, 7]
                ],
                [
                    [9, 7, 2, 7, 3],
                    [0, 3, 7, 4, 7],
                    [0, 3, 7, 4, 7]
                ]

            ],
            dtype=torch.float32
        )

        mask = torch.tensor(
            [
                [1, 1, 0],
                [1, 0, 0]
            ],
            dtype=torch.uint8
        )

        dim = -2

        correct = torch.tensor(
            [
                [0, 2, 0, 0, 1],
                [9, 7, 2, 7, 3]
            ],
            dtype=torch.float32
        )

        result = pwF.masked_min_pooling(data_tensor, mask, dim)

        self.assertListEqual(result.tolist(), correct.tolist())


class MaskedMeanPoolingTestCase(unittest.TestCase):

    def test_3D(self):
        data_tensor = torch.tensor(
            [
                [
                    [8, 4, 0, 0, 1],
                    [0, 2, 6, 1, 3],
                    [0, 3, 7, 4, 7]
                ],
                [
                    [9, 7, 2, 7, 3],
                    [0, 3, 7, 4, 7],
                    [0, 3, 7, 4, 7]
                ]
            ],
            dtype=torch.float32
        )

        mask = torch.tensor(
            [
                [
                    [0, 0, 0, 0, 1],
                    [1, 1, 0, 1, 1],
                    [0, 0, 1, 1, 1]
                ],
                [
                    [1, 0, 1, 1, 1],
                    [0, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1]
                ]
            ],
            dtype=torch.uint8
        )

        dim = -2

        correct = torch.tensor(
            [
                [0, 2, 7, (4 + 1) / 2., (1 + 3 + 7) / 3.],
                [9, 3, (2 + 7) / 2., (7 + 4 + 4) / 3., (3 + 7 + 7) / 3.]
            ],
            dtype=torch.float32
        )

        result = pwF.masked_mean_pooling(data_tensor, mask, dim)

        np.testing.assert_almost_equal(result.tolist(), correct.tolist(), 5)

    def test_3D_broadcasting_mask(self):
        data_tensor = torch.tensor(
            [
                [
                    [8, 4, 0, 0, 1],
                    [0, 2, 6, 1, 3],
                    [0, 3, 7, 4, 7]
                ],
                [
                    [9, 7, 2, 7, 3],
                    [0, 3, 7, 4, 7],
                    [0, 3, 7, 4, 7]
                ]
            ],
            dtype=torch.float32
        )

        mask = torch.tensor(
            [
                [1, 1, 0],
                [1, 0, 0]
            ],
            dtype=torch.uint8
        )

        dim = -2

        correct = torch.tensor(
            [
                [(8 + 0) / 2, (4 + 2) / 2, (0 + 6) / 2, (0 + 1) / 2, (1 + 3) / 2],
                [9, 7, 2, 7, 3]
            ],
            dtype=torch.float32
        )

        result = pwF.masked_mean_pooling(data_tensor, mask, dim)

        np.testing.assert_almost_equal(result.tolist(), correct.tolist(), 5)


class GetFirstNonMaskedElementTestCase(unittest.TestCase):

    def test_3D_data_dim_2_end_padded(self):
        data_tensor = torch.tensor(
            [
                [
                    [8, 4, 0, 0, 1],
                    [0, 2, 6, 1, 3],
                    [0, 3, 7, 4, 7]
                ],
                [
                    [9, 7, 2, 7, 3],
                    [2, 3, 4, 2, 1],
                    [0, 3, 7, 4, 7]
                ]
            ],
            dtype=torch.float32
        )

        lengths_tensor = torch.tensor(
            [
                [1, 2, 3],
                [2, 3, 4]
            ]
        )

        dim = 2
        is_end_padded = True

        correct = torch.tensor(
            [
                [8, 0, 0],
                [9, 2, 0]
            ],
            dtype=torch.float32
        )

        result = pwF.get_first_non_masked_element(data_tensor, lengths_tensor, dim, is_end_padded)

        self.assertListEqual(result.tolist(), correct.tolist())

    def test_3D_data_dim_1_end_padded(self):
        data_tensor = torch.tensor(
            [
                [
                    [8, 4, 0, 0, 1],
                    [0, 2, 6, 1, 3],
                    [0, 3, 7, 4, 7]
                ],
                [
                    [9, 7, 2, 7, 3],
                    [2, 3, 4, 2, 1],
                    [0, 3, 7, 4, 7]
                ]
            ],
            dtype=torch.float32
        )

        lengths_tensor = torch.tensor([1, 2])

        dim = 1
        is_end_padded = True

        correct = torch.tensor(
            [
                [8, 4, 0, 0, 1],
                [9, 7, 2, 7, 3]
            ],
            dtype=torch.float32
        )

        result = pwF.get_first_non_masked_element(data_tensor, lengths_tensor, dim, is_end_padded)

        self.assertListEqual(result.tolist(), correct.tolist())

    def test_3D_data_dim_2_start_padded(self):
        data_tensor = torch.tensor(
            [
                [
                    [8, 4, 0, 0, 1],
                    [0, 2, 6, 1, 3],
                    [0, 3, 7, 4, 7]
                ],
                [
                    [9, 7, 2, 7, 3],
                    [2, 3, 4, 2, 1],
                    [0, 3, 7, 4, 7]
                ]
            ],
            dtype=torch.float32
        )

        lengths_tensor = torch.tensor(
            [
                [1, 2, 3],
                [2, 3, 4]
            ]
        )

        dim = 2
        is_end_padded = False

        correct = torch.tensor(
            [
                [1, 1, 7],
                [7, 4, 3]
            ],
            dtype=torch.float32
        )

        result = pwF.get_first_non_masked_element(data_tensor, lengths_tensor, dim, is_end_padded)

        self.assertListEqual(result.tolist(), correct.tolist())

    def test_3D_data_dim_1_start_padded(self):
        data_tensor = torch.tensor(
            [
                [
                    [8, 4, 0, 0, 1],
                    [0, 2, 6, 1, 3],
                    [0, 3, 7, 4, 7]
                ],
                [
                    [9, 7, 2, 7, 3],
                    [2, 3, 4, 2, 1],
                    [0, 3, 7, 4, 7]
                ]
            ],
            dtype=torch.float32
        )

        lengths_tensor = torch.tensor([1, 2])

        dim = 1
        is_end_padded = False

        correct = torch.tensor(
            [
                [0, 3, 7, 4, 7],
                [2, 3, 4, 2, 1]
            ],
            dtype=torch.float32
        )

        result = pwF.get_first_non_masked_element(data_tensor, lengths_tensor, dim, is_end_padded)

        self.assertListEqual(result.tolist(), correct.tolist())


class GetLastNonMaskedElementTestCase(unittest.TestCase):

    def test_3D_data_dim_2_end_padded(self):
        data_tensor = torch.tensor(
            [
                [
                    [8, 4, 0, 0, 1],
                    [0, 2, 6, 1, 3],
                    [0, 3, 7, 4, 7]
                ],
                [
                    [9, 7, 2, 7, 3],
                    [2, 3, 4, 2, 1],
                    [0, 3, 7, 4, 7]
                ]
            ],
            dtype=torch.float32
        )

        lengths_tensor = torch.tensor(
            [
                [1, 2, 3],
                [2, 3, 4]
            ]
        )

        dim = 2
        is_end_padded = True

        correct = torch.tensor(
            [
                [8, 2, 7],
                [7, 4, 4]
            ],
            dtype=torch.float32
        )

        result = pwF.get_last_non_masked_element(data_tensor, lengths_tensor, dim, is_end_padded)

        self.assertListEqual(result.tolist(), correct.tolist())

    def test_3D_data_dim_1_end_padded(self):
        data_tensor = torch.tensor(
            [
                [
                    [8, 4, 0, 0, 1],
                    [0, 2, 6, 1, 3],
                    [0, 3, 7, 4, 7]
                ],
                [
                    [9, 7, 2, 7, 3],
                    [2, 3, 4, 2, 1],
                    [0, 3, 7, 4, 7]
                ]
            ],
            dtype=torch.float32
        )

        lengths_tensor = torch.tensor([1, 2])

        dim = 1
        is_end_padded = True

        correct = torch.tensor(
            [
                [8, 4, 0, 0, 1],
                [2, 3, 4, 2, 1]
            ],
            dtype=torch.float32
        )

        result = pwF.get_last_non_masked_element(data_tensor, lengths_tensor, dim, is_end_padded)

        self.assertListEqual(result.tolist(), correct.tolist())

    def test_3D_data_dim_2_start_padded(self):
        data_tensor = torch.tensor(
            [
                [
                    [8, 4, 0, 0, 1],
                    [0, 2, 6, 1, 3],
                    [0, 3, 7, 4, 7]
                ],
                [
                    [9, 7, 2, 7, 3],
                    [2, 3, 4, 2, 1],
                    [0, 3, 7, 4, 7]
                ]

            ],
            dtype=torch.float32
        )

        lengths_tensor = torch.tensor(
            [
                [1, 2, 3],
                [2, 3, 4]
            ]
        )

        dim = 2
        is_end_padded = False

        correct = torch.tensor(
            [
                [1, 3, 7],
                [3, 1, 7]
            ],
            dtype=torch.float32
        )

        result = pwF.get_last_non_masked_element(data_tensor, lengths_tensor, dim, is_end_padded)

        self.assertListEqual(result.tolist(), correct.tolist())

    def test_3D_data_dim_1_start_padded(self):
        data_tensor = torch.tensor(
            [
                [
                    [8, 4, 0, 0, 1],
                    [0, 2, 6, 1, 3],
                    [0, 3, 7, 4, 7]
                ],
                [
                    [9, 7, 2, 7, 3],
                    [2, 3, 4, 2, 1],
                    [0, 3, 7, 4, 7]
                ]
            ],
            dtype=torch.float32
        )

        lengths_tensor = torch.tensor([1, 2])

        dim = 1
        is_end_padded = False

        correct = torch.tensor(
            [
                [0, 3, 7, 4, 7],
                [0, 3, 7, 4, 7]
            ],
            dtype=torch.float32
        )

        result = pwF.get_last_non_masked_element(data_tensor, lengths_tensor, dim, is_end_padded)

        self.assertListEqual(result.tolist(), correct.tolist())


class GetLastStateOfRNNTestCase(unittest.TestCase):

    def test_unidirectional_end_padded(self):
        rnn_out = torch.tensor(
            [
                [
                    [8, 4, 0, 0, 1, 2],
                    [0, 2, 6, 1, 3, 3],
                    [0, 3, 7, 4, 7, 4]
                ],
                [
                    [9, 7, 2, 7, 3, 8],
                    [2, 3, 4, 2, 1, 9],
                    [0, 3, 7, 4, 7, 1]
                ]
            ],
            dtype=torch.float32
        )

        batch_sequence_lengths = torch.tensor([1, 2])

        is_bidirectional = False
        is_end_padded = True

        correct = torch.tensor(
            [
                [8, 4, 0, 0, 1, 2],
                [2, 3, 4, 2, 1, 9]
            ],
            dtype=torch.float32
        )

        result = pwF.get_last_state_of_rnn(rnn_out, batch_sequence_lengths, is_bidirectional, is_end_padded)

        self.assertListEqual(result.tolist(), correct.tolist())

    def test_unidirectional_start_padded(self):
        rnn_out = torch.tensor(
            [
                [
                    [8, 4, 0, 0, 1, 2],
                    [0, 2, 6, 1, 3, 3],
                    [0, 3, 7, 4, 7, 4]
                ],
                [
                    [9, 7, 2, 7, 3, 8],
                    [2, 3, 4, 2, 1, 9],
                    [0, 3, 7, 4, 7, 1]
                ]
            ],
            dtype=torch.float32
        )

        batch_sequence_lengths = torch.tensor([1, 2])

        is_bidirectional = False
        is_end_padded = False

        correct = torch.tensor(
            [
                [0, 3, 7, 4, 7, 4],
                [0, 3, 7, 4, 7, 1]
            ],
            dtype=torch.float32
        )

        result = pwF.get_last_state_of_rnn(rnn_out, batch_sequence_lengths, is_bidirectional, is_end_padded)

        self.assertListEqual(result.tolist(), correct.tolist())

    def test_bidirectional_end_padded(self):
        rnn_out = torch.tensor(
            [
                [
                    [8, 4, 0, 0, 1, 2],
                    [0, 2, 6, 1, 3, 3],
                    [0, 3, 7, 4, 7, 4]
                ],
                [
                    [9, 7, 2, 7, 3, 8],
                    [2, 3, 4, 2, 1, 9],
                    [0, 3, 7, 4, 7, 1]
                ]
            ],
            dtype=torch.float32
        )

        batch_sequence_lengths = torch.tensor([1, 2])

        is_bidirectional = True
        is_end_padded = True

        correct = torch.tensor(
            [
                [8, 4, 0, 0, 1, 2],
                [2, 3, 4, 7, 3, 8]
            ],
            dtype=torch.float32
        )

        result = pwF.get_last_state_of_rnn(rnn_out, batch_sequence_lengths, is_bidirectional, is_end_padded)

        self.assertListEqual(result.tolist(), correct.tolist())

    def test_bidirectional_start_padded(self):
        rnn_out = torch.tensor(
            [
                [
                    [8, 4, 0, 0, 1, 2],
                    [0, 2, 6, 1, 3, 3],
                    [0, 3, 7, 4, 7, 4]
                ],
                [
                    [9, 7, 2, 7, 3, 8],
                    [2, 3, 4, 2, 1, 9],
                    [0, 3, 7, 4, 7, 1]
                ]
            ],
            dtype=torch.float32
        )

        batch_sequence_lengths = torch.tensor([1, 2])

        is_bidirectional = True
        is_end_padded = False

        correct = torch.tensor(
            [
                [0, 3, 7, 4, 7, 4],
                [0, 3, 7, 2, 1, 9]
            ],
            dtype=torch.float32
        )

        result = pwF.get_last_state_of_rnn(rnn_out, batch_sequence_lengths, is_bidirectional, is_end_padded)

        self.assertListEqual(result.tolist(), correct.tolist())


class PadTestCase(unittest.TestCase):

    def test_dim_2_pad_at_start(self):
        data_tensor = torch.tensor(
            [
                [
                    [8, 4, 0, 0, 1, 2],
                    [0, 2, 6, 1, 3, 3],
                    [0, 3, 7, 4, 7, 4]
                ],
                [
                    [9, 7, 2, 7, 3, 8],
                    [2, 3, 4, 2, 1, 9],
                    [0, 3, 7, 4, 7, 1]
                ]
            ],
            dtype=torch.float32
        )

        pad_size = 2
        dim = 2
        pad_at_end = False

        correct = torch.tensor(
            [
                [
                    [0, 0, 8, 4, 0, 0, 1, 2],
                    [0, 0, 0, 2, 6, 1, 3, 3],
                    [0, 0, 0, 3, 7, 4, 7, 4]
                ],
                [
                    [0, 0, 9, 7, 2, 7, 3, 8],
                    [0, 0, 2, 3, 4, 2, 1, 9],
                    [0, 0, 0, 3, 7, 4, 7, 1]
                ]
            ],
            dtype=torch.float32
        )

        result = pwF.pad(data_tensor, pad_size, dim, pad_at_end)

        self.assertListEqual(result.tolist(), correct.tolist())

    def test_dim_2_pad_at_end(self):
        data_tensor = torch.tensor(
            [
                [
                    [8, 4, 0, 0, 1, 2],
                    [0, 2, 6, 1, 3, 3],
                    [0, 3, 7, 4, 7, 4]
                ],
                [
                    [9, 7, 2, 7, 3, 8],
                    [2, 3, 4, 2, 1, 9],
                    [0, 3, 7, 4, 7, 1]
                ]
            ],
            dtype=torch.float32
        )

        pad_size = 2
        dim = 2
        pad_at_end = True

        correct = torch.tensor(
            [
                [
                    [8, 4, 0, 0, 1, 2, 0, 0],
                    [0, 2, 6, 1, 3, 3, 0, 0],
                    [0, 3, 7, 4, 7, 4, 0, 0]
                ],
                [
                    [9, 7, 2, 7, 3, 8, 0, 0],
                    [2, 3, 4, 2, 1, 9, 0, 0],
                    [0, 3, 7, 4, 7, 1, 0, 0]
                ]
            ],
            dtype=torch.float32
        )

        result = pwF.pad(data_tensor, pad_size, dim, pad_at_end)

        self.assertListEqual(result.tolist(), correct.tolist())

    def test_dim_1_pad_at_start(self):
        data_tensor = torch.tensor(
            [
                [
                    [8, 4, 0, 0, 1, 2],
                    [0, 2, 6, 1, 3, 3],
                    [0, 3, 7, 4, 7, 4]
                ],
                [
                    [9, 7, 2, 7, 3, 8],
                    [2, 3, 4, 2, 1, 9],
                    [0, 3, 7, 4, 7, 1]
                ]
            ],
            dtype=torch.float32
        )

        pad_size = 2
        dim = 1
        pad_at_end = False

        correct = torch.tensor(
            [
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [8, 4, 0, 0, 1, 2],
                    [0, 2, 6, 1, 3, 3],
                    [0, 3, 7, 4, 7, 4]
                ],
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [9, 7, 2, 7, 3, 8],
                    [2, 3, 4, 2, 1, 9],
                    [0, 3, 7, 4, 7, 1]
                ]
            ],
            dtype=torch.float32
        )

        result = pwF.pad(data_tensor, pad_size, dim, pad_at_end)

        self.assertListEqual(result.tolist(), correct.tolist())

    def test_dim_1_pad_at_end(self):
        data_tensor = torch.tensor(
            [
                [
                    [8, 4, 0, 0, 1, 2],
                    [0, 2, 6, 1, 3, 3],
                    [0, 3, 7, 4, 7, 4]
                ],
                [
                    [9, 7, 2, 7, 3, 8],
                    [2, 3, 4, 2, 1, 9],
                    [0, 3, 7, 4, 7, 1]
                ]
            ],
            dtype=torch.float32
        )

        pad_size = 2
        dim = 1
        pad_at_end = True

        correct = torch.tensor(
            [
                [
                    [8, 4, 0, 0, 1, 2],
                    [0, 2, 6, 1, 3, 3],
                    [0, 3, 7, 4, 7, 4],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]
                ],
                [
                    [9, 7, 2, 7, 3, 8],
                    [2, 3, 4, 2, 1, 9],
                    [0, 3, 7, 4, 7, 1],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]
                ]
            ],
            dtype=torch.float32
        )

        result = pwF.pad(data_tensor, pad_size, dim, pad_at_end)

        self.assertListEqual(result.tolist(), correct.tolist())


class SameDropoutTestCase(unittest.TestCase):

    def test_model_training(self):
        data_tensor = torch.tensor(
            [
                [
                    [8, 4, 0, 0, 1, 2],
                    [9, 7, 2, 7, 3, 8],
                    [8, 4, 0, 0, 1, 2]
                ],

                [
                    [8, 4, 0, 0, 1, 2],
                    [9, 7, 2, 7, 3, 8],
                    [8, 4, 0, 0, 1, 2]
                ]
            ],
            dtype=torch.float32
        )

        dropout_p = 0.9
        dim = 0
        is_model_training = True

        result = pwF.same_dropout(data_tensor, dropout_p, dim, is_model_training)

        self.assertListEqual(result[0].tolist(), result[1].tolist())

    def test_model_not_training(self):
        data_tensor = torch.tensor(
            [
                [
                    [8, 4, 0, 0, 1, 2],
                    [9, 7, 2, 7, 3, 8],
                    [8, 4, 0, 0, 1, 2]
                ],

                [
                    [8, 4, 0, 0, 1, 2],
                    [9, 7, 2, 7, 3, 8],
                    [8, 4, 0, 0, 1, 2]
                ]
            ],
            dtype=torch.float32
        )

        dropout_p = 0.9
        dim = 0
        is_model_training = False

        result = pwF.same_dropout(data_tensor, dropout_p, dim, is_model_training)

        self.assertListEqual(result.tolist(), data_tensor.tolist())


class SubTensorDropoutTestCase(unittest.TestCase):

    def test_small_p_model_training(self):
        data_tensor = torch.tensor(
            [
                [
                    [8, 4, 0, 0, 1, 2],
                    [9, 7, 2, 7, 3, 8],
                    [8, 4, 0, 0, 1, 2]
                ],

                [
                    [8, 4, 0, 0, 1, 2],
                    [9, 7, 2, 7, 3, 8],
                    [8, 4, 0, 0, 1, 2]
                ]
            ],
            dtype=torch.float32
        )

        dropout_p = 0.0000001
        dim = 0
        is_model_training = True

        result = pwF.sub_tensor_dropout(data_tensor, dropout_p, dim, is_model_training)

        self.assertListEqual(result[0].tolist(), result[1].tolist())

    def test_large_p_model_training(self):
        data_tensor = torch.tensor(
            [
                [
                    [8, 4, 0, 0, 1, 2],
                    [9, 7, 2, 7, 3, 8],
                    [8, 4, 0, 0, 1, 2]
                ],

                [
                    [8, 4, 0, 0, 1, 2],
                    [9, 7, 2, 7, 3, 8],
                    [8, 4, 0, 0, 1, 2]
                ]
            ],
            dtype=torch.float32
        )

        dropout_p = 0.9999999
        dim = 0
        is_model_training = True

        result = pwF.sub_tensor_dropout(data_tensor, dropout_p, dim, is_model_training)

        self.assertListEqual(result[0].tolist(), result[1].tolist())

    def test_model_not_training(self):
        data_tensor = torch.tensor(
            [
                [
                    [8, 4, 0, 0, 1, 2],
                    [9, 7, 2, 7, 3, 8],
                    [8, 4, 0, 0, 1, 2]
                ],

                [
                    [8, 4, 0, 0, 1, 2],
                    [9, 7, 2, 7, 3, 8],
                    [8, 4, 0, 0, 1, 2]
                ]
            ],
            dtype=torch.float32
        )

        dropout_p = 0.9
        dim = 0
        is_model_training = False

        result = pwF.sub_tensor_dropout(data_tensor, dropout_p, dim, is_model_training)

        self.assertListEqual(result.tolist(), data_tensor.tolist())
