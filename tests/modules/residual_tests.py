import torch
import unittest
import numpy as np

from unittest.mock import MagicMock

from pytorch_wrapper import modules


class ResidualTestCase(unittest.TestCase):

    def test_execution(self):
        x = torch.tensor([
            [1, 2, 3],
            [6, 4, 3],
            [1, 2, 3]
        ], dtype=torch.float32
        )

        mocked_return_value = torch.tensor([
            [1, 2, 3],
            [6, 4, 3],
            [1, 2, 3]
        ], dtype=torch.float32
        )

        correct = x + mocked_return_value

        mocked_module = MagicMock(return_value=mocked_return_value)

        layer = modules.Residual(mocked_module)

        result = layer(x)

        np.testing.assert_almost_equal(result.tolist(), correct.tolist(), 5)
