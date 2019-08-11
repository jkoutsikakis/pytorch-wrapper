import unittest
import hyperopt

from unittest.mock import MagicMock

from pytorch_wrapper.tuner import Tuner


class TunerTestCase(unittest.TestCase):

    def test_execution(self):

        hyper_parameter_generators = {
            'param1': hyperopt.hp.choice('param1', [32, 64]),
            'param2': hyperopt.hp.choice('param2', [0.01, 0.001, 0.0001]),
            'param3': hyperopt.hp.choice('param3', [0, 0.0001, 0.001])
        }

        step_function = MagicMock()
        step_function.side_effect = [10, 9, 1, 5, 30, 2, 10, 2, 3, 1]

        algorithm = hyperopt.tpe.suggest
        fit_iterations = 10

        tuner = Tuner(hyper_parameter_generators, step_function, algorithm, fit_iterations)
        points = tuner.run()

        self.assertEqual(step_function.call_count, 10)
        self.assertEqual(len(points), 10)
