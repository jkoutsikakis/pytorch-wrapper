import numpy as np

from abc import ABC, abstractmethod
from sklearn import metrics

from . import functional as pwF


class AbstractEvaluatorResults(ABC):
    """
    Objects of derives classes encapsulate results of an evaluation metric.
    """

    @abstractmethod
    def is_better_than(self, other_results_object):
        """
        Compares these results with the results of another object.

        :param other_results_object: Object of the same class.
        """

        pass

    @abstractmethod
    def compare_to(self, other_results_object):
        """
        Compares these results with the results of another object.

        :param other_results_object: Object of the same class.
        """

        pass

    @abstractmethod
    def __str__(self):
        pass

    def __repr__(self):
        return self.__str__()


class GenericEvaluatorResults(AbstractEvaluatorResults):
    """
    Generic evaluator results.
    """

    def __init__(self, score, label='score', score_format='%f', is_max_better=True):
        """
        :param score: Numeric value that represents the score.
        :param label: String used in the str representation.
        :param score_format: Format String used in the str representation.
        :param is_max_better: Flag that signifies if larger means better.
        """

        super(GenericEvaluatorResults, self).__init__()
        self._score = score
        self._label = label
        self._score_format = score_format
        self._is_max_better = is_max_better

    @property
    def score(self):
        return self._score

    @property
    def is_max_better(self):
        return self._is_max_better

    def is_better_than(self, other_results_object):
        if other_results_object is None:
            return True

        if self._is_max_better:
            return self.compare_to(other_results_object) > 0
        else:
            return self.compare_to(other_results_object) < 0

    def compare_to(self, other_results_object):
        return self._score - other_results_object.score

    def __str__(self):
        return (self._label + ': ' + self._score_format) % self._score


class AbstractEvaluator(ABC):
    """
    Objects of derived classes are used to evaluate a model on a dataset using a specific metric.
    """

    def __init__(self):
        self.reset()

    @abstractmethod
    def reset(self):
        """
        (Re)initializes the object. Called at the beginning of the evaluation step.
        """

        pass

    @abstractmethod
    def step(self, output, batch, last_activation=None):
        """
        Gathers information needed for performance measurement about a single batch. Called after each batch in the
        evaluation step.

        :param output: Output of the model.
        :param batch: Dict that contains all information needed for a single batch by the evaluator.
        :param last_activation: The last activation of the model. Some losses work with logits and as such the last
            activation might not be performed inside the model's forward method.
        """

        pass

    @abstractmethod
    def calculate(self):
        """
        Called after all batches have been processed. Calculates the metric.

        :return: AbstractEvaluatorResults object.
        """

        pass

    def calculate_at_once(self, output, dataset, last_activation=None):
        """
        Calculates the metric at once for the whole dataset.

        :param output:  Output of the model.
        :param dataset: Dict that contains all information needed for a dataset by the evaluator.
        :param last_activation: The last activation of the model. Some losses work with logits and as such the last
            activation might not be performed inside the model's forward method.
        :return: AbstractEvaluatorResults object.
        """

        self.reset()
        self.step(output, dataset, last_activation)
        return self.calculate()


class GenericPointWiseLossEvaluator(AbstractEvaluator):
    """
    Adapter that uses an object of a class derived from AbstractLossWrapper to calculate the loss during evaluation.
    """

    def __init__(self, loss_wrapper, label='loss', score_format='%f', batch_target_key='target'):
        """
        :param loss_wrapper: AbstractLossWrapper object that calculates the loss.
        :param label: Str used as label during printing of the loss.
        :param score_format: Format used for str representation of the loss.
        :param batch_target_key: Key where the dict (batch) contains the target values.
        """

        super(GenericPointWiseLossEvaluator, self).__init__()
        self._loss_wrapper = loss_wrapper
        self._label = label
        self._score_format = score_format
        self._batch_target_key = batch_target_key
        self.reset()

    def reset(self):
        self._loss = 0
        self._examples_nb = 0

    def step(self, output, batch, last_activation=None):
        current_loss = self._loss_wrapper.calculate_loss(output, batch, None, last_activation).item()
        self._loss += current_loss * batch[self._batch_target_key].shape[0]
        self._examples_nb += batch[self._batch_target_key].shape[0]

    def calculate(self):
        return GenericEvaluatorResults(self._loss / self._examples_nb, self._label, self._score_format,
                                       is_max_better=False)


class AccuracyEvaluator(AbstractEvaluator):
    """
    Accuracy evaluator.
    """

    def __init__(self, threshold=0.5, model_output_key=None, batch_target_key='target'):
        """
        :param threshold: Threshold above which an example is considered positive.
        :param model_output_key: Key where the dict returned by the model contains the actual predictions. Leave None
            if the model returns only the predictions.
        :param batch_target_key: Key where the dict (batch) contains the target values.
        """

        super(AccuracyEvaluator, self).__init__()
        self._threshold = threshold
        self._model_output_key = model_output_key
        self._batch_target_key = batch_target_key
        self.reset()

    def reset(self):
        self._outputs = []
        self._targets = []

    def step(self, output, batch, last_activation=None):
        if self._model_output_key is not None:
            output = output[self._model_output_key]
        if last_activation is not None:
            output = last_activation(output)
        self._outputs.extend(output.tolist())
        self._targets.extend(batch[self._batch_target_key].tolist())

    def calculate(self):
        predictions = np.array(self._outputs) > self._threshold
        targets = np.array(self._targets) > self._threshold
        correct = (predictions == targets).sum()
        return GenericEvaluatorResults(100.0 * correct / predictions.size, 'acc', '%5.2f%%', is_max_better=True)


class MultiClassAccuracyEvaluator(AbstractEvaluator):
    """
    Multi-Class Accuracy evaluator.
    """

    def __init__(self, model_output_key=None, batch_target_key='target'):
        """
        :param model_output_key: Key where the dict returned by the model contains the actual predictions. Leave None
            if the model returns only the predictions.
        :param batch_target_key: Key where the dict (batch) contains the target values.
        """

        super(MultiClassAccuracyEvaluator, self).__init__()
        self._model_output_key = model_output_key
        self._batch_target_key = batch_target_key
        self.reset()

    def reset(self):
        self._outputs = []
        self._targets = []

    def step(self, output, batch, last_activation=None):
        if self._model_output_key is not None:
            output = output[self._model_output_key]
        self._outputs.extend(output.tolist())
        self._targets.extend(batch[self._batch_target_key].tolist())

    def calculate(self):
        predictions = np.array(self._outputs).argmax(axis=-1)
        correct = (predictions == self._targets).sum()
        return GenericEvaluatorResults(100.0 * correct / predictions.shape[0], 'acc', '%5.2f%%', is_max_better=True)


class AUROCEvaluator(AbstractEvaluator):
    """
    AUROC evaluator.
    """

    def __init__(self, model_output_key=None, batch_target_key='target', average='macro', target_threshold=0.5):
        """
        :param model_output_key: Key where the dict returned by the model contains the actual predictions. Leave None
            if the model returns only the predictions.
        :param batch_target_key: Key where the dict (batch) contains the target values.
        :param average: Type ['macro' or 'micro'] of averaging performed on the results in case of multi-label task.
        """

        super(AUROCEvaluator, self).__init__()
        self._model_output_key = model_output_key
        self._batch_target_key = batch_target_key
        self._average = average
        self._target_threshold = target_threshold
        self.reset()

    def reset(self):
        self._outputs = []
        self._targets = []

    def step(self, output, batch, last_activation=None):
        if self._model_output_key is not None:
            output = output[self._model_output_key]
        if last_activation is not None:
            output = last_activation(output)
        self._outputs.extend(output.tolist())
        self._targets.extend(batch[self._batch_target_key].tolist())

    def calculate(self):
        return GenericEvaluatorResults(metrics.roc_auc_score(
            y_score=np.array(self._outputs, dtype='float32'),
            y_true=np.array(self._targets) > self._target_threshold,
            average=self._average
        ), 'auroc', '%5.4f', is_max_better=True)


class PrecisionEvaluator(AbstractEvaluator):
    """
    Precision evaluator.
    """

    def __init__(self, threshold=0.5, model_output_key=None, batch_target_key='target', average='binary'):
        """
        :param threshold: Threshold above which an example is considered positive.
        :param model_output_key: Key where the dict returned by the model contains the actual predictions. Leave None
            if the model returns only the predictions.
        :param batch_target_key: Key where the dict (batch) contains the target values.
        :param average: Type ['binary', 'macro' or 'micro'] of averaging performed on the results.
        """

        super(PrecisionEvaluator, self).__init__()
        self._threshold = threshold
        self._model_output_key = model_output_key
        self._batch_target_key = batch_target_key
        self._average = average
        self.reset()

    def reset(self):
        self._outputs = []
        self._targets = []

    def step(self, output, batch, last_activation=None):
        if self._model_output_key is not None:
            output = output[self._model_output_key]
        if last_activation is not None:
            output = last_activation(output)
        self._outputs.extend(output.tolist())
        self._targets.extend(batch[self._batch_target_key].tolist())

    def calculate(self):
        return GenericEvaluatorResults(metrics.precision_score(
            y_pred=np.array(self._outputs) > self._threshold,
            y_true=np.array(self._targets) > self._threshold,
            average=self._average
        ), self._average + '-precision', '%5.4f', is_max_better=True)


class MultiClassPrecisionEvaluator(AbstractEvaluator):
    """
    Multi-Class Precision evaluator.
    """

    def __init__(self, model_output_key=None, batch_target_key='target', average='macro'):
        """
        :param model_output_key: Key where the dict returned by the model contains the actual predictions. Leave None
            if the model returns only the predictions.
        :param batch_target_key: Key where the dict (batch) contains the target values.
        :param average: Type ['macro' or 'micro'] of averaging performed on the results.
        """

        super(MultiClassPrecisionEvaluator, self).__init__()
        self._model_output_key = model_output_key
        self._batch_target_key = batch_target_key
        self._average = average
        self.reset()

    def reset(self):
        self._outputs = []
        self._targets = []

    def step(self, output, batch, last_activation=None):
        if self._model_output_key is not None:
            output = output[self._model_output_key]
        if last_activation is not None:
            output = last_activation(output)
        self._outputs.extend(output.tolist())
        self._targets.extend(batch[self._batch_target_key].tolist())

    def calculate(self):
        return GenericEvaluatorResults(metrics.precision_score(
            y_pred=np.array(self._outputs).argmax(axis=-1),
            y_true=np.array(self._targets),
            average=self._average
        ), self._average + '-precision', '%5.4f', is_max_better=True)


class RecallEvaluator(AbstractEvaluator):
    """
    Recall evaluator.
    """

    def __init__(self, threshold=0.5, model_output_key=None, batch_target_key='target', average='binary'):
        """
        :param threshold: Threshold above which an example is considered positive.
        :param model_output_key: Key where the dict returned by the model contains the actual predictions. Leave None
            if the model returns only the predictions.
        :param batch_target_key: Key where the dict (batch) contains the target values.
        :param average: Type ['binary', 'macro' or 'micro'] of averaging performed on the results.
        """

        super(RecallEvaluator, self).__init__()
        self._threshold = threshold
        self._model_output_key = model_output_key
        self._batch_target_key = batch_target_key
        self._average = average
        self.reset()

    def reset(self):
        self._outputs = []
        self._targets = []

    def step(self, output, batch, last_activation=None):
        if self._model_output_key is not None:
            output = output[self._model_output_key]
        if last_activation is not None:
            output = last_activation(output)
        self._outputs.extend(output.tolist())
        self._targets.extend(batch[self._batch_target_key].tolist())

    def calculate(self):
        return GenericEvaluatorResults(metrics.recall_score(
            y_pred=np.array(self._outputs) > self._threshold,
            y_true=np.array(self._targets) > self._threshold,
            average=self._average
        ), self._average + '-recall', '%5.4f', is_max_better=True)


class MultiClassRecallEvaluator(AbstractEvaluator):
    """
    Multi-Class Recall evaluator.
    """

    def __init__(self, model_output_key=None, batch_target_key='target', average='macro'):
        """
        :param model_output_key: Key where the dict returned by the model contains the actual predictions. Leave None
            if the model returns only the predictions.
        :param batch_target_key: Key where the dict (batch) contains the target values.
        :param average: Type ['macro' or 'micro'] of averaging performed on the results.
        """

        super(MultiClassRecallEvaluator, self).__init__()
        self._model_output_key = model_output_key
        self._batch_target_key = batch_target_key
        self._average = average
        self.reset()

    def reset(self):
        self._outputs = []
        self._targets = []

    def step(self, output, batch, last_activation=None):
        if self._model_output_key is not None:
            output = output[self._model_output_key]
        if last_activation is not None:
            output = last_activation(output)
        self._outputs.extend(output.tolist())
        self._targets.extend(batch[self._batch_target_key].tolist())

    def calculate(self):
        return GenericEvaluatorResults(metrics.recall_score(
            y_pred=np.array(self._outputs).argmax(axis=-1),
            y_true=np.array(self._targets),
            average=self._average
        ), self._average + '-recall', '%5.4f', is_max_better=True)


class F1Evaluator(AbstractEvaluator):
    """
    F1 evaluator.
    """

    def __init__(self, threshold=0.5, model_output_key=None, batch_target_key='target', average='binary'):
        """
        :param threshold: Threshold above which an example is considered positive.
        :param model_output_key: Key where the dict returned by the model contains the actual predictions. Leave None
            if the model returns only the predictions.
        :param batch_target_key: Key where the dict (batch) contains the target values.
        :param average: Type ['binary', 'macro' or 'micro'] of averaging performed on the results.
        """

        super(F1Evaluator, self).__init__()
        self._threshold = threshold
        self._model_output_key = model_output_key
        self._batch_target_key = batch_target_key
        self._average = average
        self.reset()

    def reset(self):
        self._outputs = []
        self._targets = []

    def step(self, output, batch, last_activation=None):
        if self._model_output_key is not None:
            output = output[self._model_output_key]
        if last_activation is not None:
            output = last_activation(output)
        self._outputs.extend(output.tolist())
        self._targets.extend(batch[self._batch_target_key].tolist())

    def calculate(self):
        return GenericEvaluatorResults(metrics.f1_score(
            y_pred=np.array(self._outputs) > self._threshold,
            y_true=np.array(self._targets) > self._threshold,
            average=self._average
        ), self._average + '-f1', '%5.4f', is_max_better=True)


class MultiClassF1Evaluator(AbstractEvaluator):
    """
    Multi-Class F1 evaluator.
    """

    def __init__(self, model_output_key=None, batch_target_key='target', average='macro'):
        """
        :param model_output_key: Key where the dict returned by the model contains the actual predictions. Leave None
            if the model returns only the predictions.
        :param batch_target_key: Key where the dict (batch) contains the target values.
        :param average: Type ['macro' or 'micro'] of averaging performed on the results.
        """

        super(MultiClassF1Evaluator, self).__init__()
        self._model_output_key = model_output_key
        self._batch_target_key = batch_target_key
        self._average = average
        self.reset()

    def reset(self):
        self._outputs = []
        self._targets = []

    def step(self, output, batch, last_activation=None):
        if self._model_output_key is not None:
            output = output[self._model_output_key]
        if last_activation is not None:
            output = last_activation(output)
        self._outputs.extend(output.tolist())
        self._targets.extend(batch[self._batch_target_key].tolist())

    def calculate(self):
        return GenericEvaluatorResults(metrics.f1_score(
            y_pred=np.array(self._outputs).argmax(axis=-1),
            y_true=np.array(self._targets),
            average=self._average
        ), self._average + '-f1', '%5.4f', is_max_better=True)


class SequenceLabelingEvaluatorWrapper(AbstractEvaluator):
    """
    Adapter that wraps a pointwise loss module. It is used in sequence labeling tasks in order to flat the output and
    target while discarding invalid values due to padding.
    """

    def __init__(self, evaluator, batch_input_sequence_length_idx, batch_input_key='input', model_output_key=None,
                 batch_target_key='target', end_padded=True):
        """
        :param evaluator: The evaluator.
        :param batch_input_sequence_length_idx: The index of the input list where the lengths of the sequences can be
            found.
        :param batch_input_key: Key of the Dicts returned by the Dataloader objects that corresponds to the input of the
            model.
        :param model_output_key: Key where the dict returned by the model contains the actual predictions. Leave None
            if the model returns only the predictions.
        :param batch_target_key: Key where the dict (batch) contains the target values.
        :param end_padded: Whether the sequences are end-padded.
        """

        self._evaluator = evaluator
        super(SequenceLabelingEvaluatorWrapper, self).__init__()
        self._batch_input_sequence_length_idx = batch_input_sequence_length_idx
        self._batch_input_key = batch_input_key
        self._model_output_key = model_output_key
        self._batch_target_key = batch_target_key
        self._end_padded = end_padded
        self.reset()

    def reset(self):
        self._evaluator.reset()

    def step(self, output, batch, last_activation=None):
        if self._model_output_key is not None:
            output = output[self._model_output_key]

        mask = pwF.create_mask_from_length(
            batch[self._batch_input_key][self._batch_input_sequence_length_idx].to(output.device),
            output.shape[1],
            self._end_padded
        ).view(-1)

        new_output = output.view(output.shape[0] * output.shape[1], -1).squeeze(-1)

        batch_targets = batch[self._batch_target_key]
        batch_targets = batch_targets.view(batch_targets.shape[0] * batch_targets.shape[1], -1).squeeze(-1)

        new_output = new_output[mask]
        batch_targets = batch_targets[mask]

        new_batch = {k: batch[k] for k in batch if k != self._batch_target_key}
        new_batch[self._batch_target_key] = batch_targets

        self._evaluator.step(new_output, new_batch, last_activation)

    def calculate(self):
        return self._evaluator.calculate()
