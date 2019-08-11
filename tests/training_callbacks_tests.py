import unittest

from unittest.mock import MagicMock

from pytorch_wrapper.training_callbacks import NumberOfEpochsStoppingCriterionCallback, EarlyStoppingCriterionCallback


class NumberOfEpochsStoppingCriterionCallbackTestCase(unittest.TestCase):

    def test_nb_of_epochs_10(self):
        nb_of_epochs = 10
        callback = NumberOfEpochsStoppingCriterionCallback(nb_of_epochs)

        training_context = {
            '_current_epoch': -1,
            'stop_training': False
        }

        for i in range(nb_of_epochs):
            training_context['_current_epoch'] += 1
            self.assertFalse(training_context['stop_training'])
            callback.on_epoch_end(training_context)

        self.assertTrue(training_context['stop_training'])


class EarlyStoppingCriterionCallbackCallbackTestCase(unittest.TestCase):

    def test_execution(self):
        patience = 3
        evaluation_data_loader_key = 'dl'
        evaluator_key = 'e'
        best_state_filepath = 'test_file'

        callback = EarlyStoppingCriterionCallback(patience, evaluation_data_loader_key, evaluator_key,
                                                  best_state_filepath)

        training_context = {
            'system': MagicMock(),
            '_current_epoch': -1,
            'stop_training': False,
            '_results_history': []
        }

        training_context['system'].save_model_state = MagicMock()
        training_context['system'].load_model_state = MagicMock()
        is_better_list = [True, False, False, True, False, True, False, False, False, False]

        callback.on_training_start(training_context)

        for i in range(10):
            training_context['_current_epoch'] += 1
            self.assertFalse(training_context['stop_training'])
            current_eval = MagicMock()
            current_eval.is_better_than = MagicMock(return_value=is_better_list[i])
            training_context['_results_history'].append({evaluation_data_loader_key: {evaluator_key: current_eval}})
            callback.on_evaluation_end(training_context)

        callback.on_training_end(training_context)

        self.assertTrue(training_context['stop_training'])
        self.assertEqual(training_context['system'].save_model_state.call_count, 3)
        training_context['system'].load_model_state.assert_called_once()

