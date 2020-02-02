import torch
import torch.nn.functional as F
import unittest
import numpy as np

from unittest.mock import MagicMock, patch
from torch import nn
from collections import OrderedDict

from pytorch_wrapper import system as system_module
from pytorch_wrapper.system import System


class SimpleModuleOneInput(nn.Module):

    def __init__(self):
        super(SimpleModuleOneInput, self).__init__()
        self.weights = nn.Parameter(torch.ones((2, 2)))

    def forward(self, x):
        return torch.sum(F.linear(x, self.weights), dim=-1)


class SimpleModuleOneInputOutKey(nn.Module):

    def __init__(self):
        super(SimpleModuleOneInputOutKey, self).__init__()
        self.weights = nn.Parameter(torch.ones((2, 2)))

    def forward(self, x):
        return {'out': torch.sum(F.linear(x, self.weights), dim=-1)}


class SimpleModuleTwoInputs(nn.Module):

    def __init__(self):
        super(SimpleModuleTwoInputs, self).__init__()
        self.weights = nn.Parameter(torch.ones((2, 2)))

    def forward(self, x1, x2):
        return torch.sum(F.linear(x1 + x2, self.weights), dim=-1)


class SystemGeneralTestCase(unittest.TestCase):

    def test_creation(self):
        model = MagicMock()
        model.to = MagicMock()

        device = torch.device('cpu')
        system = System(model, device=device)

        self.assertEqual(system.device, device)
        model.to.assert_called_once_with(device)

    def test_to(self):
        if not torch.cuda.is_available():
            return

        model_two_inputs = SimpleModuleTwoInputs()

        device = torch.device('cpu')
        system = System(model_two_inputs, device=device)
        device_to_move = torch.device('cuda')
        system.to(device_to_move)

        self.assertEqual(device_to_move, system.device)


class SystemPredictBatchTestCase(unittest.TestCase):

    def test_predict_batch_one_input(self):
        model_one_input = SimpleModuleOneInput()

        device = torch.device('cpu')
        system = System(model_one_input, device=device)

        batch = [torch.ones((1, 2)) * 2]
        out = system.predict_batch(batch)
        self.assertAlmostEqual(out.item(), 8)

        batch = (torch.ones((1, 2)) * 2)
        out = system.predict_batch(batch)
        self.assertAlmostEqual(out.item(), 8)

        batch = torch.ones((1, 2)) * 2
        out = system.predict_batch(batch)
        self.assertAlmostEqual(out.item(), 8)

    def test_predict_batch_two_inputs(self):
        model_two_inputs = SimpleModuleTwoInputs()

        device = torch.device('cpu')
        system = System(model_two_inputs, device=device)

        batch = [torch.ones((1, 2)) * 2, torch.ones((1, 2)) * 2]
        out = system.predict_batch(batch)
        self.assertAlmostEqual(out.item(), 16)

        batch = (torch.ones((1, 2)) * 2, torch.ones((1, 2)) * 2)
        out = system.predict_batch(batch)
        self.assertAlmostEqual(out.item(), 16)


class SystemPredictTestCase(unittest.TestCase):

    def test_defaults(self):
        model_one_input = SimpleModuleOneInput()

        device = torch.device('cpu')
        last_activation = nn.ReLU()
        system = System(model_one_input, device=device, last_activation=last_activation)

        data_loader = [
            {'input': (torch.ones((1, 2)) * 2)},
            {'input': (torch.ones((1, 2)) * -2)}
        ]

        out = system.predict(data_loader=data_loader, verbose=False)

        np.testing.assert_almost_equal(out['outputs'], [8, 0], 5)

    def test_no_activation(self):
        model_one_input = SimpleModuleOneInput()

        device = torch.device('cpu')
        last_activation = nn.ReLU()
        system = System(model_one_input, device=device, last_activation=last_activation)

        data_loader = [
            {'input': (torch.ones((1, 2)) * 2)},
            {'input': (torch.ones((1, 2)) * -2)}
        ]

        out = system.predict(data_loader=data_loader, perform_last_activation=False, verbose=False)

        np.testing.assert_almost_equal(out['outputs'], [8, -8], 5)

    def test_batch_input_key(self):
        model_one_input = SimpleModuleOneInput()

        device = torch.device('cpu')
        last_activation = nn.ReLU()
        system = System(model_one_input, device=device, last_activation=last_activation)

        data_loader = [
            {'input_test': (torch.ones((1, 2)) * 2)},
            {'input_test': (torch.ones((1, 2)) * -2)}
        ]

        out = system.predict(data_loader=data_loader, batch_input_key='input_test', verbose=False)

        np.testing.assert_almost_equal(out['outputs'], [8, 0], 5)

    def test_model_output_key(self):
        model_one_input = SimpleModuleOneInputOutKey()

        device = torch.device('cpu')
        last_activation = nn.ReLU()
        system = System(model_one_input, device=device, last_activation=last_activation)

        data_loader = [
            {'input': (torch.ones((1, 2)) * 2)},
            {'input': (torch.ones((1, 2)) * -2)}
        ]

        out = system.predict(data_loader=data_loader, model_output_key='out', verbose=False)

        np.testing.assert_almost_equal(out['outputs'], [8, 0], 5)

    def test_batch_int_id_key(self):
        model_one_input = SimpleModuleOneInput()

        device = torch.device('cpu')
        last_activation = nn.ReLU()
        system = System(model_one_input, device=device, last_activation=last_activation)

        data_loader = [
            {'input': (torch.ones((1, 2)) * 2), 'id': [0]},
            {'input': (torch.ones((1, 2)) * -2), 'id': [1]}
        ]

        out = system.predict(data_loader=data_loader, batch_id_key='id', verbose=False)

        np.testing.assert_almost_equal(out['outputs'], [8, 0], 5)
        self.assertListEqual(out['id'], [0, 1])

    def test_batch_str_id_key(self):
        model_one_input = SimpleModuleOneInput()

        device = torch.device('cpu')
        last_activation = nn.ReLU()
        system = System(model_one_input, device=device, last_activation=last_activation)

        data_loader = [
            {'input': (torch.ones((1, 2)) * 2), 'id': ['id_0']},
            {'input': (torch.ones((1, 2)) * -2), 'id': ['id_1']}
        ]

        out = system.predict(data_loader=data_loader, batch_id_key='id', verbose=False)

        np.testing.assert_almost_equal(out['outputs'], [8, 0], 5)
        self.assertListEqual(out['id'], ['id_0', 'id_1'])


class SystemSaveLoadTestCase(unittest.TestCase):

    def setUp(self):
        self.path = 'test.model'
        self.last_activation = nn.ReLU
        self.model = SimpleModuleOneInput()
        self.system = System(self.model, last_activation=self.last_activation)

    def test_save(self):
        with patch('pytorch_wrapper.system.torch.save') as torch_save:
            self.system.save(self.path)

            torch_save.assert_called_once_with({
                'model': self.model,
                'last_activation': self.last_activation
            }, self.path)

    def test_load(self):
        with patch('pytorch_wrapper.system.torch.load') as torch_load:
            torch_load.return_value = {'model': self.model, 'last_activation': self.last_activation}
            loaded_system = System.load(self.path)

            torch_load.assert_called_once_with(self.path, map_location=torch.device('cpu'))
            self.assertEqual(loaded_system.model, self.model)
            self.assertEqual(loaded_system.last_activation, self.last_activation)

    def test_save_model_state(self):
        with patch('pytorch_wrapper.system.torch.save') as torch_save:
            self.system.save_model_state(self.path)
            torch_save.assert_called_once()

    def test_load_model_state(self):
        with patch('pytorch_wrapper.system.torch.load') as torch_load:
            torch_load.return_value = self.model.state_dict()
            self.system.load_model_state(self.path)

            torch_load.assert_called_once_with(self.path, map_location=torch.device('cpu'))


class SystemEvaluateTestCase(unittest.TestCase):

    def setUp(self):
        self.model_one_input = SimpleModuleOneInput()

        device = torch.device('cpu')
        last_activation = nn.ReLU()
        self.system = System(self.model_one_input, device=device, last_activation=last_activation)

        evaluator1 = MagicMock()
        evaluator1.reset = MagicMock()
        evaluator1.step = MagicMock()
        evaluator1.calculate = MagicMock(return_value='evaluator1_res')

        evaluator2 = MagicMock()
        evaluator2.reset = MagicMock()
        evaluator2.step = MagicMock()
        evaluator2.calculate = MagicMock(return_value='evaluator2_res')

        self.evaluators = {
            'evaluator1': evaluator1,
            'evaluator2': evaluator2
        }

    def test_evaluate_defaults(self):
        data_loader = [
            {'input': (torch.ones((1, 2)) * 2)},
            {'input': (torch.ones((1, 2)) * -2)}
        ]

        results = self.system.evaluate(data_loader, self.evaluators, verbose=False)

        self.assertFalse(self.model_one_input.training)

        self.evaluators['evaluator1'].reset.assert_called_once()
        self.assertEqual(self.evaluators['evaluator1'].step.call_count, 2)
        self.evaluators['evaluator1'].calculate.assert_called_once()

        self.evaluators['evaluator2'].reset.assert_called_once()
        self.assertEqual(self.evaluators['evaluator2'].step.call_count, 2)
        self.evaluators['evaluator2'].calculate.assert_called_once()

        self.assertEqual(results['evaluator1'], 'evaluator1_res')
        self.assertEqual(results['evaluator2'], 'evaluator2_res')

    def test_evaluate_batch_input_key(self):
        data_loader = [
            {'input2': (torch.ones((1, 2)) * 2)},
            {'input2': (torch.ones((1, 2)) * -2)}
        ]

        results = self.system.evaluate(data_loader, self.evaluators, batch_input_key='input2', verbose=False)

        self.assertFalse(self.model_one_input.training)

        self.evaluators['evaluator1'].reset.assert_called_once()
        self.assertEqual(self.evaluators['evaluator1'].step.call_count, 2)
        self.evaluators['evaluator1'].calculate.assert_called_once()

        self.evaluators['evaluator2'].reset.assert_called_once()
        self.assertEqual(self.evaluators['evaluator2'].step.call_count, 2)
        self.evaluators['evaluator2'].calculate.assert_called_once()

        self.assertEqual(results['evaluator1'], 'evaluator1_res')
        self.assertEqual(results['evaluator2'], 'evaluator2_res')


class SystemPurePredictTestCase(unittest.TestCase):

    def test_defaults(self):
        model_one_input = SimpleModuleOneInput()

        device = torch.device('cpu')
        last_activation = nn.ReLU()
        system = System(model_one_input, device=device, last_activation=last_activation)

        data_loader = [
            {'input': (torch.ones((1, 2)) * 2)},
            {'input': (torch.ones((1, 2)) * -2)}
        ]

        pure_predictions = system.pure_predict(data_loader=data_loader, verbose=False)

        self.assertEqual(len(pure_predictions['batch_list']), 2)
        self.assertEqual(len(pure_predictions['batch_list']), len(pure_predictions['output_list']))
        np.testing.assert_almost_equal(
            pure_predictions['batch_list'][0]['input'].tolist(),
            data_loader[0]['input'].tolist(),
            5
        )
        np.testing.assert_almost_equal(
            pure_predictions['batch_list'][1]['input'].tolist(),
            data_loader[1]['input'].tolist(),
            5
        )

        self.assertAlmostEqual(pure_predictions['output_list'][0].item(), 8., 5)
        self.assertAlmostEqual(pure_predictions['output_list'][1].item(), -8., 5)

    def test_batch_input_key(self):
        model_one_input = SimpleModuleOneInput()

        device = torch.device('cpu')
        last_activation = nn.ReLU()
        system = System(model_one_input, device=device, last_activation=last_activation)

        data_loader = [
            {'input_test': (torch.ones((1, 2)) * 2)},
            {'input_test': (torch.ones((1, 2)) * -2)}
        ]

        pure_predictions = system.pure_predict(data_loader=data_loader, batch_input_key='input_test', verbose=False)

        self.assertEqual(len(pure_predictions['batch_list']), 2)
        self.assertEqual(len(pure_predictions['batch_list']), len(pure_predictions['output_list']))
        np.testing.assert_almost_equal(
            pure_predictions['batch_list'][0]['input_test'].tolist(),
            data_loader[0]['input_test'].tolist(),
            5
        )
        np.testing.assert_almost_equal(
            pure_predictions['batch_list'][1]['input_test'].tolist(),
            data_loader[1]['input_test'].tolist(),
            5
        )

        self.assertAlmostEqual(pure_predictions['output_list'][0].item(), 8., 5)
        self.assertAlmostEqual(pure_predictions['output_list'][1].item(), -8., 5)

    def test_false_keep_batches(self):
        model_one_input = SimpleModuleOneInput()

        device = torch.device('cpu')
        last_activation = nn.ReLU()
        system = System(model_one_input, device=device, last_activation=last_activation)

        data_loader = [
            {'input': (torch.ones((1, 2)) * 2)},
            {'input': (torch.ones((1, 2)) * -2)}
        ]

        pure_predictions = system.pure_predict(data_loader=data_loader, keep_batches=False, verbose=False)

        self.assertEqual(len(pure_predictions['output_list']), 2)
        self.assertFalse('batch_list' in pure_predictions)
        self.assertAlmostEqual(pure_predictions['output_list'][0].item(), 8., 5)
        self.assertAlmostEqual(pure_predictions['output_list'][1].item(), -8., 5)


class SystemTrainTestCase(unittest.TestCase):

    def setUp(self):
        self.callback = MagicMock()
        self.callback.on_training_star = MagicMock()
        self.callback.on_training_end = MagicMock()
        self.callback.on_evaluation_start = MagicMock()
        self.callback.on_evaluation_end = MagicMock()
        self.callback.on_epoch_start = MagicMock()
        self.callback.on_epoch_end = MagicMock()
        self.callback.on_batch_start = MagicMock()
        self.callback.on_batch_end = MagicMock()
        self.callback.post_predict = MagicMock()
        self.callback.post_loss_calculation = MagicMock()
        self.callback.post_backward_calculation = MagicMock()
        self.callback.pre_optimization_step = MagicMock()

        self.model_one_input = SimpleModuleOneInput()

        device = torch.device('cpu')
        last_activation = nn.ReLU()
        self.system = System(self.model_one_input, device=device, last_activation=last_activation)

        self.data_loader_1 = [
            {'input': (torch.ones((1, 2)) * 2)},
            {'input': (torch.ones((1, 2)) * -2)}
        ]

        self.data_loader_2 = [
            {'input': (torch.ones((1, 2)) * 3)},
            {'input': (torch.ones((1, 2)) * -3)}
        ]

        self.data_loader_3 = [
            {'input': (torch.ones((1, 2)) * 4)},
            {'input': (torch.ones((1, 2)) * -4)}
        ]

        evaluator1 = MagicMock()
        evaluator1.reset = MagicMock()
        evaluator1.step = MagicMock()
        evaluator1.calculate = MagicMock(return_value='evaluator1_res')

        evaluator2 = MagicMock()
        evaluator2.reset = MagicMock()
        evaluator2.step = MagicMock()
        evaluator2.calculate = MagicMock(return_value='evaluator2_res')

        self.loss_wrapper = MagicMock()
        self.loss_wrapper.calculate_loss = MagicMock()
        self.calculate_loss_ret_value = MagicMock()
        self.calculate_loss_ret_value.backward = MagicMock()
        self.calculate_loss_ret_value.item = MagicMock()
        self.calculate_loss_ret_value.item.return_value = 1
        self.loss_wrapper.calculate_loss.return_value = self.calculate_loss_ret_value
        self.optimizer = MagicMock()
        self.optimizer.zero_grad = MagicMock()
        self.optimizer.step = MagicMock()
        self.train_data_loader = self.data_loader_1
        self.evaluation_data_loaders = OrderedDict([
            ('d1', self.data_loader_2),
            ('d2', self.data_loader_3)
        ])
        self.batch_input_key = 'input'
        self.evaluators = OrderedDict([
            ('e1', evaluator1),
            ('e2', evaluator2)
        ])
        self.callbacks = [self.callback]
        self.gradient_accumulation_steps = 1
        self.verbose = False

        self.trainer = system_module._Trainer(
            self.system,
            self.loss_wrapper,
            self.optimizer,
            self.train_data_loader,
            self.evaluation_data_loaders,
            self.batch_input_key,
            self.evaluators,
            self.callbacks,
            self.gradient_accumulation_steps,
            self.verbose
        )

    def test_train_epoch(self):
        with patch('pytorch_wrapper.system._Trainer._train_batch') as train_batch:
            train_batch.return_value = 1
            self.trainer._train_epoch()

            self.assertEqual(train_batch.call_count, 2)
            self.assertEqual(self.trainer.training_context['_current_epoch'], 0)
            self.callback.on_epoch_start.assert_called_once()
            self.callback.on_epoch_end.assert_called_once()

    def test_train_batch(self):
        batch = self.data_loader_1[0]
        perform_opt_step = True

        with patch('pytorch_wrapper.system.System.predict_batch') as predict_batch:
            predict_batch.return_value = 1
            cur_batch_loss = self.trainer._train_batch(batch, perform_opt_step)

            self.assertEqual(self.trainer.training_context['current_batch'], None)
            self.assertEqual(self.trainer.training_context['current_output'], None)
            self.assertEqual(self.trainer.training_context['current_loss'], None)
            self.trainer.training_context['optimizer'].zero_grad.assert_called_once()
            self.trainer.training_context['optimizer'].step.assert_called_once()
            self.trainer.training_context['loss_wrapper'].calculate_loss.assert_called_once()
            self.calculate_loss_ret_value.backward.assert_called_once()
            self.callback.on_batch_start.assert_called_once()
            self.callback.post_predict.assert_called_once()
            self.callback.post_loss_calculation.assert_called_once()
            self.callback.post_backward_calculation.assert_called_once()
            self.callback.pre_optimization_step.assert_called_once()
            self.callback.on_batch_end.assert_called_once()
            self.assertEqual(cur_batch_loss, 1)

    def test_train_evaluation(self):
        with patch('pytorch_wrapper.system.System.evaluate') as evaluate:
            evaluate.side_effect = [{'e1': 'd1_e1', 'e2': 'd1_e2'}, {'e1': 'd2_e1', 'e2': 'd2_e2'}]
            self.trainer._train_evaluation()

            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual(self.trainer.training_context['_results_history'][0]['d1']['e1'], 'd1_e1')
            self.assertEqual(self.trainer.training_context['_results_history'][0]['d1']['e2'], 'd1_e2')
            self.assertEqual(self.trainer.training_context['_results_history'][0]['d2']['e1'], 'd2_e1')
            self.assertEqual(self.trainer.training_context['_results_history'][0]['d2']['e2'], 'd2_e2')
            self.callback.on_evaluation_start.assert_called_once()
            self.callback.on_evaluation_end.assert_called_once()

    def test_train(self):
        with patch('pytorch_wrapper.system._Trainer._train_epoch') as train_epoch, \
                patch('pytorch_wrapper.system._Trainer._train_evaluation') as train_evaluation:
            ret_stop_training = MagicMock(side_effect=[False, True])
            training_context = self.trainer.training_context
            self.trainer.training_context = MagicMock()
            self.trainer.training_context.__getitem__.side_effect = lambda key: training_context[key] \
                if key != 'stop_training' else ret_stop_training()

            self.trainer.run()

            self.callback.on_training_start.assert_called_once()
            self.callback.on_training_end.assert_called_once()
            train_epoch.assert_called_once()
            train_evaluation.assert_called_once()
