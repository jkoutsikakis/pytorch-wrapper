import unittest
import torch
import numpy as np

from unittest.mock import MagicMock
from sklearn import metrics

from pytorch_wrapper import evaluators


class GenericEvaluatorResultsTestCase(unittest.TestCase):

    def test_max_is_better(self):
        score_1 = 1.
        score_2 = 2.

        er1 = evaluators.GenericEvaluatorResults(score_1, is_max_better=True)
        er2 = evaluators.GenericEvaluatorResults(score_2, is_max_better=True)

        self.assertTrue(er2.is_better_than(er1))
        self.assertFalse(er1.is_better_than(er2))
        self.assertAlmostEqual(er2.compare_to(er1), 1)
        self.assertAlmostEqual(er1.compare_to(er2), -1)

    def test_min_is_better(self):
        score_1 = 1.
        score_2 = 2.

        er1 = evaluators.GenericEvaluatorResults(score_1, is_max_better=False)
        er2 = evaluators.GenericEvaluatorResults(score_2, is_max_better=False)

        self.assertFalse(er2.is_better_than(er1))
        self.assertTrue(er1.is_better_than(er2))
        self.assertAlmostEqual(er2.compare_to(er1), 1)
        self.assertAlmostEqual(er1.compare_to(er2), -1)


class GenericPointWiseLossEvaluatorTestCase(unittest.TestCase):

    def test_correct_loss_calculation(self):
        mocked_loss = MagicMock()
        mocked_loss.item = MagicMock(return_value=10)

        loss_wrapper = MagicMock()
        loss_wrapper.calculate_loss = MagicMock(return_value=mocked_loss)

        evaluator = evaluators.GenericPointWiseLossEvaluator(loss_wrapper,
                                                             label='loss',
                                                             score_format='%f',
                                                             batch_target_key='target')

        output = MagicMock()
        batch = {'target': MagicMock()}
        batch['target'].shape = [5]
        evaluator.step(output, batch)
        batch['target'].shape = [8]
        evaluator.step(output, batch)

        res = evaluator.calculate()

        self.assertAlmostEqual(res.score, 10)


class AccuracyEvaluatorTestCase(unittest.TestCase):

    def test_correct_score_calculation_binary(self):
        evaluator = evaluators.AccuracyEvaluator(threshold=0.5,
                                                 model_output_key=None,
                                                 batch_target_key='target')

        output = torch.tensor([0.9, 0.2, 0.7, 0.4], dtype=torch.float32)
        batch = {'target': torch.tensor([1, 1, 0, 0], dtype=torch.float32)}
        evaluator.step(output, batch)

        output = torch.tensor([0.3, 0.7, 0.9], dtype=torch.float32)
        batch = {'target': torch.tensor([1, 1, 0], dtype=torch.float32)}
        evaluator.step(output, batch)

        res = evaluator.calculate()

        self.assertAlmostEqual(res.score, 100 * 3. / 7)

    def test_correct_score_calculation_multi_label(self):
        evaluator = evaluators.AccuracyEvaluator(threshold=0.5,
                                                 model_output_key=None,
                                                 batch_target_key='target')

        output = torch.tensor([[0.7, 0.4], [0.8, 0.3], [0.7, 0.6], [0.2, 0.8]], dtype=torch.float32)
        batch = {'target': torch.tensor([[1, 1], [0, 1], [1, 0], [0, 1]], dtype=torch.float32)}
        evaluator.step(output, batch)

        output = torch.tensor([[0.8, 0.3]], dtype=torch.float32)
        batch = {'target': torch.tensor([[1, 1]], dtype=torch.float32)}
        evaluator.step(output, batch)

        res = evaluator.calculate()

        self.assertAlmostEqual(res.score, 100 * 5. / 10)


class MultiClassAccuracyEvaluatorTestCase(unittest.TestCase):

    def test_correct_score_calculation(self):
        evaluator = evaluators.MultiClassAccuracyEvaluator(model_output_key=None,
                                                           batch_target_key='target')

        output = torch.tensor([[0.5, 0.1, 0.4], [0.3, 0.3, 0.4], [0.5, 0.5, 0.0]], dtype=torch.float32)
        batch = {'target': torch.tensor([0, 2, 2], dtype=torch.float32)}
        evaluator.step(output, batch)

        output = torch.tensor([[0.1, 0.1, 0.8]], dtype=torch.float32)
        batch = {'target': torch.tensor([2], dtype=torch.float32)}
        evaluator.step(output, batch)

        res = evaluator.calculate()

        self.assertAlmostEqual(res.score, 100 * 3. / 4)


class AUROCEvaluatorTestCase(unittest.TestCase):

    def test_correct_score_calculation_binary(self):
        evaluator = evaluators.AUROCEvaluator(model_output_key=None,
                                              batch_target_key='target',
                                              average='macro')

        output = torch.tensor([0.9, 0.2, 0.8, 0.3], dtype=torch.float32)
        batch = {'target': torch.tensor([1, 1, 0, 0], dtype=torch.float32)}
        evaluator.step(output, batch)

        output = torch.tensor([0.2, 0.98, 0.76], dtype=torch.float32)
        batch = {'target': torch.tensor([1, 1, 0], dtype=torch.float32)}
        evaluator.step(output, batch)

        res = evaluator.calculate()

        self.assertAlmostEqual(res.score, 0.5)

    def test_correct_score_calculation_multi_label_macro(self):
        evaluator = evaluators.AUROCEvaluator(model_output_key=None,
                                              batch_target_key='target',
                                              average='macro')

        output = torch.tensor([[0.6, 0.2], [0.7, 0.2], [0.6, 0.6], [0.3, 0.55]], dtype=torch.float32)
        batch = {'target': torch.tensor([[1, 1], [0, 1], [1, 0], [0, 1]], dtype=torch.float32)}
        evaluator.step(output, batch)

        output = torch.tensor([[0.6, 0.4]], dtype=torch.float32)
        batch = {'target': torch.tensor([[1, 1]], dtype=torch.float32)}
        evaluator.step(output, batch)

        res = evaluator.calculate()

        lable1_score = metrics.roc_auc_score(y_score=np.array([0.6, 0.7, 0.6, 0.3, 0.6]),
                                             y_true=np.array([1, 0, 1, 0, 1]))

        label2_score = metrics.roc_auc_score(y_score=np.array([0.2, 0.2, 0.6, 0.55, 0.4]),
                                             y_true=np.array([1, 1, 0, 1, 1]))

        correct = (lable1_score + label2_score) / 2.

        self.assertAlmostEqual(res.score, correct)

    def test_correct_score_calculation_multi_label_micro(self):
        evaluator = evaluators.AUROCEvaluator(model_output_key=None,
                                              batch_target_key='target',
                                              average='micro')

        output = torch.tensor([[0.6, 0.2], [0.7, 0.2], [0.6, 0.6], [0.3, 0.55]], dtype=torch.float32)
        batch = {'target': torch.tensor([[1, 1], [0, 1], [1, 0], [0, 1]], dtype=torch.float32)}
        evaluator.step(output, batch)

        output = torch.tensor([[0.6, 0.4]], dtype=torch.float32)
        batch = {'target': torch.tensor([[1, 1]], dtype=torch.float32)}
        evaluator.step(output, batch)

        res = evaluator.calculate()

        correct = metrics.roc_auc_score(y_score=np.array([0.6, 0.7, 0.6, 0.3, 0.6, 0.2, 0.2, 0.6, 0.55, 0.4]),
                                        y_true=np.array([1, 0, 1, 0, 1, 1, 1, 0, 1, 1]))

        self.assertAlmostEqual(res.score, correct)


class PrecisionEvaluatorTestCase(unittest.TestCase):

    def test_correct_score_calculation_binary(self):
        evaluator = evaluators.PrecisionEvaluator(model_output_key=None,
                                                  batch_target_key='target',
                                                  average='binary')

        output = torch.tensor([0.9, 0.2, 0.8, 0.3], dtype=torch.float32)
        batch = {'target': torch.tensor([1, 1, 0, 0], dtype=torch.float32)}
        evaluator.step(output, batch)

        output = torch.tensor([0.2, 0.98, 0.76], dtype=torch.float32)
        batch = {'target': torch.tensor([1, 1, 0], dtype=torch.float32)}
        evaluator.step(output, batch)

        res = evaluator.calculate()

        self.assertAlmostEqual(res.score, 2. / 4)

    def test_correct_score_calculation_multi_label_macro(self):
        evaluator = evaluators.PrecisionEvaluator(model_output_key=None,
                                                  batch_target_key='target',
                                                  average='macro')

        output = torch.tensor([[0.6, 0.2], [0.7, 0.2], [0.6, 0.6], [0.3, 0.55]], dtype=torch.float32)
        batch = {'target': torch.tensor([[1, 1], [0, 1], [1, 0], [0, 1]], dtype=torch.float32)}
        evaluator.step(output, batch)

        output = torch.tensor([[0.6, 0.4]], dtype=torch.float32)
        batch = {'target': torch.tensor([[1, 1]], dtype=torch.float32)}
        evaluator.step(output, batch)

        res = evaluator.calculate()

        lable1_score = metrics.precision_score(y_pred=np.array([0.6, 0.7, 0.6, 0.3, 0.6]) > 0.5,
                                               y_true=np.array([1, 0, 1, 0, 1]))

        label2_score = metrics.precision_score(y_pred=np.array([0.2, 0.2, 0.6, 0.55, 0.4]) > 0.5,
                                               y_true=np.array([1, 1, 0, 1, 1]))

        correct = (lable1_score + label2_score) / 2.

        self.assertAlmostEqual(res.score, correct)

    def test_correct_score_calculation_multi_label_micro(self):
        evaluator = evaluators.PrecisionEvaluator(model_output_key=None,
                                                  batch_target_key='target',
                                                  average='micro')

        output = torch.tensor([[0.6, 0.2], [0.7, 0.2], [0.6, 0.6], [0.3, 0.55]], dtype=torch.float32)
        batch = {'target': torch.tensor([[1, 1], [0, 1], [1, 0], [0, 1]], dtype=torch.float32)}
        evaluator.step(output, batch)

        output = torch.tensor([[0.6, 0.4]], dtype=torch.float32)
        batch = {'target': torch.tensor([[1, 1]], dtype=torch.float32)}
        evaluator.step(output, batch)

        res = evaluator.calculate()

        correct = metrics.precision_score(y_pred=np.array([0.6, 0.7, 0.6, 0.3, 0.6, 0.2, 0.2, 0.6, 0.55, 0.4]) > 0.5,
                                          y_true=np.array([1, 0, 1, 0, 1, 1, 1, 0, 1, 1]))

        self.assertAlmostEqual(res.score, correct)


class MultiClassPrecisionEvaluatorTestCase(unittest.TestCase):

    def test_correct_score_calculation_macro(self):
        evaluator = evaluators.MultiClassPrecisionEvaluator(model_output_key=None,
                                                            batch_target_key='target',
                                                            average='macro')

        output = torch.tensor([[0.5, 0.1, 0.4], [0.3, 0.3, 0.4], [0.5, 0.5, 0.0]], dtype=torch.float32)
        batch = {'target': torch.tensor([0, 2, 2], dtype=torch.float32)}
        evaluator.step(output, batch)

        output = torch.tensor([[0.1, 0.1, 0.8]], dtype=torch.float32)
        batch = {'target': torch.tensor([2], dtype=torch.float32)}
        evaluator.step(output, batch)

        res = evaluator.calculate()

        correct = metrics.precision_score(y_pred=np.array([0, 2, 0, 2]), y_true=np.array([0, 2, 2, 2]), average='macro')

        self.assertAlmostEqual(res.score, correct)

    def test_correct_score_calculation_micro(self):
        evaluator = evaluators.MultiClassPrecisionEvaluator(model_output_key=None,
                                                            batch_target_key='target',
                                                            average='micro')

        output = torch.tensor([[0.5, 0.1, 0.4], [0.3, 0.3, 0.4], [0.5, 0.5, 0.0]], dtype=torch.float32)
        batch = {'target': torch.tensor([0, 2, 2], dtype=torch.float32)}
        evaluator.step(output, batch)

        output = torch.tensor([[0.1, 0.1, 0.8]], dtype=torch.float32)
        batch = {'target': torch.tensor([2], dtype=torch.float32)}
        evaluator.step(output, batch)

        res = evaluator.calculate()

        correct = metrics.precision_score(y_pred=np.array([0, 2, 0, 2]), y_true=np.array([0, 2, 2, 2]), average='micro')

        self.assertAlmostEqual(res.score, correct)


class RecallEvaluatorTestCase(unittest.TestCase):

    def test_correct_score_calculation_binary(self):
        evaluator = evaluators.RecallEvaluator(model_output_key=None,
                                               batch_target_key='target',
                                               average='binary')

        output = torch.tensor([0.9, 0.2, 0.8, 0.3], dtype=torch.float32)
        batch = {'target': torch.tensor([1, 1, 0, 0], dtype=torch.float32)}
        evaluator.step(output, batch)

        output = torch.tensor([0.2, 0.98, 0.76], dtype=torch.float32)
        batch = {'target': torch.tensor([1, 1, 0], dtype=torch.float32)}
        evaluator.step(output, batch)

        res = evaluator.calculate()

        self.assertAlmostEqual(res.score, 0.5)

    def test_correct_score_calculation_multi_label_macro(self):
        evaluator = evaluators.RecallEvaluator(model_output_key=None,
                                               batch_target_key='target',
                                               average='macro')

        output = torch.tensor([[0.6, 0.2], [0.7, 0.2], [0.6, 0.6], [0.3, 0.55]], dtype=torch.float32)
        batch = {'target': torch.tensor([[1, 1], [0, 1], [1, 0], [0, 1]], dtype=torch.float32)}
        evaluator.step(output, batch)

        output = torch.tensor([[0.6, 0.4]], dtype=torch.float32)
        batch = {'target': torch.tensor([[1, 1]], dtype=torch.float32)}
        evaluator.step(output, batch)

        res = evaluator.calculate()

        lable1_score = metrics.recall_score(y_pred=np.array([0.6, 0.7, 0.6, 0.3, 0.6]) > 0.5,
                                            y_true=np.array([1, 0, 1, 0, 1]))

        label2_score = metrics.recall_score(y_pred=np.array([0.2, 0.2, 0.6, 0.55, 0.4]) > 0.5,
                                            y_true=np.array([1, 1, 0, 1, 1]))

        correct = (lable1_score + label2_score) / 2.

        self.assertAlmostEqual(res.score, correct)

    def test_correct_score_calculation_multi_label_micro(self):
        evaluator = evaluators.RecallEvaluator(model_output_key=None,
                                               batch_target_key='target',
                                               average='micro')

        output = torch.tensor([[0.6, 0.2], [0.7, 0.2], [0.6, 0.6], [0.3, 0.55]], dtype=torch.float32)
        batch = {'target': torch.tensor([[1, 1], [0, 1], [1, 0], [0, 1]], dtype=torch.float32)}
        evaluator.step(output, batch)

        output = torch.tensor([[0.6, 0.4]], dtype=torch.float32)
        batch = {'target': torch.tensor([[1, 1]], dtype=torch.float32)}
        evaluator.step(output, batch)

        res = evaluator.calculate()

        correct = metrics.recall_score(y_pred=np.array([0.6, 0.7, 0.6, 0.3, 0.6, 0.2, 0.2, 0.6, 0.55, 0.4]) > 0.5,
                                       y_true=np.array([1, 0, 1, 0, 1, 1, 1, 0, 1, 1]))

        self.assertAlmostEqual(res.score, correct)


class MultiClassRecallEvaluatorTestCase(unittest.TestCase):

    def test_correct_score_calculation_macro(self):
        evaluator = evaluators.MultiClassRecallEvaluator(model_output_key=None,
                                                         batch_target_key='target',
                                                         average='macro')

        output = torch.tensor([[0.5, 0.1, 0.4], [0.3, 0.3, 0.4], [0.5, 0.5, 0.0]], dtype=torch.float32)
        batch = {'target': torch.tensor([0, 2, 2], dtype=torch.float32)}
        evaluator.step(output, batch)

        output = torch.tensor([[0.1, 0.1, 0.8]], dtype=torch.float32)
        batch = {'target': torch.tensor([2], dtype=torch.float32)}
        evaluator.step(output, batch)

        res = evaluator.calculate()

        correct = metrics.recall_score(y_pred=np.array([0, 2, 0, 2]), y_true=np.array([0, 2, 2, 2]), average='macro')

        self.assertAlmostEqual(res.score, correct)

    def test_correct_score_calculation_micro(self):
        evaluator = evaluators.MultiClassRecallEvaluator(model_output_key=None,
                                                         batch_target_key='target',
                                                         average='micro')

        output = torch.tensor([[0.5, 0.1, 0.4], [0.3, 0.3, 0.4], [0.5, 0.5, 0.0]], dtype=torch.float32)
        batch = {'target': torch.tensor([0, 2, 2], dtype=torch.float32)}
        evaluator.step(output, batch)

        output = torch.tensor([[0.1, 0.1, 0.8]], dtype=torch.float32)
        batch = {'target': torch.tensor([2], dtype=torch.float32)}
        evaluator.step(output, batch)

        res = evaluator.calculate()

        correct = metrics.recall_score(y_pred=np.array([0, 2, 0, 2]), y_true=np.array([0, 2, 2, 2]), average='micro')

        self.assertAlmostEqual(res.score, correct)


class F1EvaluatorTestCase(unittest.TestCase):

    def test_correct_score_calculation_binary(self):
        evaluator = evaluators.F1Evaluator(model_output_key=None,
                                           batch_target_key='target',
                                           average='binary')

        output = torch.tensor([0.9, 0.2, 0.8, 0.3], dtype=torch.float32)
        batch = {'target': torch.tensor([1, 1, 0, 0], dtype=torch.float32)}
        evaluator.step(output, batch)

        output = torch.tensor([0.2, 0.98, 0.76], dtype=torch.float32)
        batch = {'target': torch.tensor([1, 1, 0], dtype=torch.float32)}
        evaluator.step(output, batch)

        res = evaluator.calculate()

        self.assertAlmostEqual(res.score, 0.5)

    def test_correct_score_calculation_multi_label_macro(self):
        evaluator = evaluators.F1Evaluator(model_output_key=None,
                                           batch_target_key='target',
                                           average='macro')

        output = torch.tensor([[0.6, 0.2], [0.7, 0.2], [0.6, 0.6], [0.3, 0.55]], dtype=torch.float32)
        batch = {'target': torch.tensor([[1, 1], [0, 1], [1, 0], [0, 1]], dtype=torch.float32)}
        evaluator.step(output, batch)

        output = torch.tensor([[0.6, 0.4]], dtype=torch.float32)
        batch = {'target': torch.tensor([[1, 1]], dtype=torch.float32)}
        evaluator.step(output, batch)

        res = evaluator.calculate()

        lable1_score = metrics.f1_score(y_pred=np.array([0.6, 0.7, 0.6, 0.3, 0.6]) > 0.5,
                                        y_true=np.array([1, 0, 1, 0, 1]))

        label2_score = metrics.f1_score(y_pred=np.array([0.2, 0.2, 0.6, 0.55, 0.4]) > 0.5,
                                        y_true=np.array([1, 1, 0, 1, 1]))

        correct = (lable1_score + label2_score) / 2.

        self.assertAlmostEqual(res.score, correct)

    def test_correct_score_calculation_multi_label_micro(self):
        evaluator = evaluators.F1Evaluator(model_output_key=None,
                                           batch_target_key='target',
                                           average='micro')

        output = torch.tensor([[0.6, 0.2], [0.7, 0.2], [0.6, 0.6], [0.3, 0.55]], dtype=torch.float32)
        batch = {'target': torch.tensor([[1, 1], [0, 1], [1, 0], [0, 1]], dtype=torch.float32)}
        evaluator.step(output, batch)

        output = torch.tensor([[0.6, 0.4]], dtype=torch.float32)
        batch = {'target': torch.tensor([[1, 1]], dtype=torch.float32)}
        evaluator.step(output, batch)

        res = evaluator.calculate()

        correct = metrics.f1_score(y_pred=np.array([0.6, 0.7, 0.6, 0.3, 0.6, 0.2, 0.2, 0.6, 0.55, 0.4]) > 0.5,
                                   y_true=np.array([1, 0, 1, 0, 1, 1, 1, 0, 1, 1]))

        self.assertAlmostEqual(res.score, correct)


class MultiClassF1EvaluatorTestCase(unittest.TestCase):

    def test_correct_score_calculation_macro(self):
        evaluator = evaluators.MultiClassF1Evaluator(model_output_key=None,
                                                     batch_target_key='target',
                                                     average='macro')

        output = torch.tensor([[0.5, 0.1, 0.4], [0.3, 0.3, 0.4], [0.5, 0.5, 0.0]], dtype=torch.float32)
        batch = {'target': torch.tensor([0, 2, 2], dtype=torch.float32)}
        evaluator.step(output, batch)

        output = torch.tensor([[0.1, 0.1, 0.8]], dtype=torch.float32)
        batch = {'target': torch.tensor([2], dtype=torch.float32)}
        evaluator.step(output, batch)

        res = evaluator.calculate()

        correct = metrics.f1_score(y_pred=np.array([0, 2, 0, 2]), y_true=np.array([0, 2, 2, 2]), average='macro')

        self.assertAlmostEqual(res.score, correct)

    def test_correct_score_calculation_micro(self):
        evaluator = evaluators.MultiClassF1Evaluator(model_output_key=None,
                                                     batch_target_key='target',
                                                     average='micro')

        output = torch.tensor([[0.5, 0.1, 0.4], [0.3, 0.3, 0.4], [0.5, 0.5, 0.0]], dtype=torch.float32)
        batch = {'target': torch.tensor([0, 2, 2], dtype=torch.float32)}
        evaluator.step(output, batch)

        output = torch.tensor([[0.1, 0.1, 0.8]], dtype=torch.float32)
        batch = {'target': torch.tensor([2], dtype=torch.float32)}
        evaluator.step(output, batch)

        res = evaluator.calculate()

        correct = metrics.f1_score(y_pred=np.array([0, 2, 0, 2]), y_true=np.array([0, 2, 2, 2]), average='micro')

        self.assertAlmostEqual(res.score, correct)


class TokenLabelingEvaluatorWrapperTestCase(unittest.TestCase):

    def test_f1_binary(self):
        bi_sequence_len_idx = 1
        batch_input_key = 'input'
        model_output_key = None
        batch_target_key = 'target'
        end_padded = True

        wrapped_evaluator = evaluators.F1Evaluator(model_output_key=model_output_key,
                                                   batch_target_key=batch_target_key,
                                                   average='binary')
        evaluator = evaluators.TokenLabelingEvaluatorWrapper(evaluator=wrapped_evaluator,
                                                             batch_input_sequence_length_idx=bi_sequence_len_idx,
                                                             batch_input_key=batch_input_key,
                                                             model_output_key=model_output_key,
                                                             batch_target_key=batch_target_key,
                                                             end_padded=end_padded)

        output = torch.tensor([[0.9, 0.2, -2.], [0.8, 0.3, -2.]])
        batch = {'target': torch.tensor([[1., 1., -1.], [0., 0., -1.]]),
                 'input': [None, torch.tensor([2, 2], dtype=torch.int)]}
        evaluator.step(output, batch)

        output = torch.tensor([[0.2, 0.98, -2.], [0.76, -2, -2.]])
        batch = {'target': torch.tensor([[1., 1., -1.], [0., -1, -1.]]),
                 'input': [None, torch.tensor([2, 1], dtype=torch.int)]}
        evaluator.step(output, batch)

        res = evaluator.calculate()

        self.assertAlmostEqual(res.score, 0.5)

    def test_f1_multi_label_macro(self):
        bi_sequence_len_idx = 1
        batch_input_key = 'input'
        model_output_key = None
        batch_target_key = 'target'
        end_padded = True
        wrapped_evaluator = evaluators.F1Evaluator(model_output_key=model_output_key,
                                                   batch_target_key=batch_target_key,
                                                   average='macro')

        evaluator = evaluators.TokenLabelingEvaluatorWrapper(evaluator=wrapped_evaluator,
                                                             batch_input_sequence_length_idx=bi_sequence_len_idx,
                                                             batch_input_key=batch_input_key,
                                                             model_output_key=model_output_key,
                                                             batch_target_key=batch_target_key,
                                                             end_padded=end_padded)

        output = torch.tensor([[[0.6, 0.2], [0.7, 0.2], [-2., -2.]], [[0.6, 0.6], [0.3, 0.55], [-2., -2.]]],
                              dtype=torch.float32)
        batch = {'target': torch.tensor([[[1, 1], [0, 1], [-1, -1]], [[1, 0], [0, 1], [-1, -1]]], dtype=torch.float32),
                 'input': [None, torch.tensor([2, 2], dtype=torch.int)]}
        evaluator.step(output, batch)

        output = torch.tensor([[[0.6, 0.4]]], dtype=torch.float32)
        batch = {'target': torch.tensor([[[1, 1]]], dtype=torch.float32),
                 'input': [None, torch.tensor([1], dtype=torch.int)]}
        evaluator.step(output, batch)

        res = evaluator.calculate()

        lable1_score = metrics.f1_score(y_pred=np.array([0.6, 0.7, 0.6, 0.3, 0.6]) > 0.5,
                                        y_true=np.array([1, 0, 1, 0, 1]))

        label2_score = metrics.f1_score(y_pred=np.array([0.2, 0.2, 0.6, 0.55, 0.4]) > 0.5,
                                        y_true=np.array([1, 1, 0, 1, 1]))

        correct = (lable1_score + label2_score) / 2.

        self.assertAlmostEqual(res.score, correct)

    def test_f1_multi_class_macro(self):
        bi_sequence_len_idx = 1
        batch_input_key = 'input'
        model_output_key = None
        batch_target_key = 'target'
        end_padded = True
        wrapped_evaluator = evaluators.MultiClassF1Evaluator(model_output_key=model_output_key,
                                                             batch_target_key=batch_target_key,
                                                             average='macro')

        evaluator = evaluators.TokenLabelingEvaluatorWrapper(evaluator=wrapped_evaluator,
                                                             batch_input_sequence_length_idx=bi_sequence_len_idx,
                                                             batch_input_key=batch_input_key,
                                                             model_output_key=model_output_key,
                                                             batch_target_key=batch_target_key,
                                                             end_padded=end_padded)

        output = torch.tensor([[[0.5, 0.1, 0.4], [0.3, 0.3, 0.4]], [[0.6, 0.4, 0.0], [-2., -2., -2.]]],
                              dtype=torch.float32)
        batch = {'target': torch.tensor([[0, 2], [2, -1]], dtype=torch.float32),
                 'input': [None, torch.tensor([2, 1], dtype=torch.int)]}
        evaluator.step(output, batch)

        output = torch.tensor([[[0.1, 0.1, 0.8]]], dtype=torch.float32)
        batch = {'target': torch.tensor([[2]], dtype=torch.float32),
                 'input': [None, torch.tensor([1], dtype=torch.int)]}
        evaluator.step(output, batch)

        res = evaluator.calculate()

        correct = metrics.f1_score(y_pred=np.array([0, 2, 0, 2]), y_true=np.array([0, 2, 2, 2]), average='macro')

        self.assertAlmostEqual(res.score, correct)
