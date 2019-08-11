from __future__ import print_function

import pprint
import pickle

from abc import ABC, abstractmethod
from hyperopt import fmin, Trials
from tqdm.auto import tqdm


class AbstractTuner(ABC):
    """
    Objects of derived classes are used to tune a model using the Hyperopt library.
    """

    def __init__(self, hyper_parameter_generators, algorithm, fit_iterations):
        """
        :param hyper_parameter_generators: Dict containing a hyperopt hyper-parameter generator for each hyper-parameter
            (e.g. {'batch_size': hp.choice('batch_size', [32, 64])})
        :param algorithm: Hyperopt's tuning algorithm (e.g. hyperopt.rand.suggest, hyperopt.tpe.suggest).
        :param fit_iterations: Number of trials.
        """

        super(AbstractTuner, self).__init__()
        self._hyper_parameter_generators = hyper_parameter_generators
        self._algorithm = algorithm
        self._fit_iterations = fit_iterations
        self._current_trials_object = None
        self._trials_save_path = None

    def run(self, trials_load_path=None, trials_save_path=None):
        """
        Initiates the tuning algorithm.

        :param trials_load_path: Path of a Trials object to load at the beginning of the tuning algorithm. If None the
            tuning algorithm will start from scratch.
        :param trials_save_path: Path where to save the Trials object after each iteration. If None the Trials object
            will not be saved.
        :return: A sorted list of tuples [ (loss, {parameters}), ... ].
        """

        self._current_iteration = 0
        self._points = []

        self._trials_save_path = trials_save_path

        if trials_load_path is None:
            self._current_trials_object = Trials()
        else:
            with open(trials_load_path, 'rb') as fr:
                self._current_trials_object = pickle.load(fr)

        _ = fmin(
            fn=self._step_wrapper_fn,
            space=self._hyper_parameter_generators,
            algo=self._algorithm,
            trials=self._current_trials_object,
            max_evals=self._fit_iterations
        )

        self._points.sort(key=lambda x: x[0])

        return self._points

    def _step_wrapper_fn(self, hyper_parameters):
        """
        Wraps the user-defined _step method and stores information regarding the current iteration.

        :param hyper_parameters: Dict containing the chosen hyper-parameters for the current iteration.
        :return: Numeric value representing the loss returned from the user-defined _step method.
        """

        self._current_iteration += 1
        tqdm.write('Iteration: {0:d}/{1:d}'.format(self._current_iteration, self._fit_iterations))
        self._print_hyper_parameters(hyper_parameters)
        loss = self._step(hyper_parameters)
        self._points.append((loss, hyper_parameters))

        if self._trials_save_path is not None:
            with open(self._trials_save_path, 'wb') as fw:
                pickle.dump(self._current_trials_object, fw)

        return loss

    @staticmethod
    def _print_hyper_parameters(hyper_parameters):
        """
        Prints parameters.

        :param hyper_parameters: Dict with the hyper parameters.
        """

        tqdm.write('-' * 80)
        tqdm.write('Hyper-Parameters')
        tqdm.write('-' * 80)
        tqdm.write(pprint.pformat(hyper_parameters))
        tqdm.write('-' * 80)

    @abstractmethod
    def _step(self, hyper_parameters):
        """
        Creates and evaluates a model using the hyper-parameters provided.

        :param hyper_parameters: Dict containing the chosen hyper-parameters for the current iteration. The key for each
            hyper-parameter is the same as its corresponding generator.
        :return: Numeric value representing the loss of the current iteration.
        """

        pass


class Tuner(AbstractTuner):
    """
    Objects of this class are used to tune a model using the Hyperopt library.
    """

    def __init__(self, hyper_parameter_generators, step_function, algorithm, fit_iterations):
        """
        :param hyper_parameter_generators: Dict containing a hyperopt hyper-parameter generator for each hyper-parameter
            (e.g. {'batch_size': hp.choice('batch_size', [32, 64])})
        :param step_function: callable that creates and evaluates a model using the provided hyper-parameters. A dict
            will be provided as an argument containing the chosen hyper-parameters for the current iteration. The key
            for each hyper-parameter is the same as its corresponding generator. It must return a numeric value
            representing the loss of the current iteration.
        :param algorithm: Hyperopt's tuning algorithm (e.g. hyperopt.rand.suggest, hyperopt.tpe.suggest).
        :param fit_iterations: Number of trials.
        """
        
        super(Tuner, self).__init__(hyper_parameter_generators, algorithm, fit_iterations)
        self._step_function = step_function

    def _step(self, hyper_parameters):
        """
        Creates and evaluates a model using the hyper-parameters provided.

        :param hyper_parameters: Dict containing the chosen hyper-parameters for the current iteration. The key for each
            hyper-parameter is the same as its corresponding generator.
        :return: Numeric value representing the loss of the current iteration.
        """

        return self._step_function(hyper_parameters)
