from abc import ABC, abstractmethod

from . import functional as pwF


class AbstractLossWrapper(ABC):
    """
    Objects of derived classes are used to wrap a loss module providing an interface used by the System class.
    """

    @abstractmethod
    def calculate_loss(self, batch, output, training_context, last_activation=None):
        """
        Calculates the loss for a single batch.

        :param batch: Dict that contains all information needed by the loss wrapper.
        :param output: Output of the model.
        :param training_context: Dict containing information regarding the training process.
        :param last_activation: Last activation provided to the System.
        :return: Output of the loss function/module.
        """
        pass


class GenericPointWiseLossWrapper(AbstractLossWrapper):
    """
    Adapter that wraps a pointwise loss module.
    """

    def __init__(self, loss, model_output_key=None, batch_target_key='target', perform_last_activation=False):
        """
        :param loss: Loss module.
        :param model_output_key: Key where the dict returned by the model contains the actual predictions. Leave None
            if the model returns only the predictions.
        :param batch_target_key: Key where the dict (batch) contains the target values.
        :param perform_last_activation: Whether to perform the last_activation.
        """
        super(GenericPointWiseLossWrapper, self).__init__()
        self._loss = loss
        self._model_output_key = model_output_key
        self._batch_target_key = batch_target_key
        self._perform_last_activation = perform_last_activation

    def calculate_loss(self, output, batch, training_context, last_activation=None):

        if self._model_output_key is not None:
            output = output[self._model_output_key]

        if last_activation is not None and self._perform_last_activation:
            output = last_activation(output)

        batch_targets = batch[self._batch_target_key].to(output.device)

        return self._loss(output, batch_targets)


class SequenceLabelingGenericPointWiseLossWrapper(AbstractLossWrapper):
    """
    Adapter that wraps a pointwise loss module. It is used in sequence labeling tasks in order to flat the output and
    target while discarding invalid values due to padding.
    """

    def __init__(self, loss, batch_input_sequence_length_idx, batch_input_key='input', model_output_key=None,
                 batch_target_key='target', perform_last_activation=False, end_padded=True):
        """
        :param loss: Loss module.
        :param batch_input_sequence_length_idx: The index of the input list where the lengths of the sequences can be
            found.
        :param batch_input_key: Key of the Dicts returned by the Dataloader objects that corresponds to the input of the
            model.
        :param model_output_key: Key where the dict returned by the model contains the actual predictions. Leave None
            if the model returns only the predictions.
        :param batch_target_key: Key where the dict (batch) contains the target values.
        :param perform_last_activation: Whether to perform the last_activation.
        :param end_padded: Whether the sequences are end-padded.
        """
        super(SequenceLabelingGenericPointWiseLossWrapper, self).__init__()
        self._loss = loss
        self._batch_input_sequence_length_idx = batch_input_sequence_length_idx
        self._batch_input_key = batch_input_key
        self._model_output_key = model_output_key
        self._batch_target_key = batch_target_key
        self._perform_last_activation = perform_last_activation
        self._end_padded = end_padded

    def calculate_loss(self, output, batch, training_context, last_activation=None):

        if self._model_output_key is not None:
            output = output[self._model_output_key]

        if last_activation is not None and self._perform_last_activation:
            output = last_activation(output)

        mask = pwF.create_mask_from_length(
            batch[self._batch_input_key][self._batch_input_sequence_length_idx].to(output.device),
            output.shape[1],
            self._end_padded
        ).view(-1)

        output = output.view(output.shape[0] * output.shape[1], -1).squeeze(-1)

        batch_targets = batch[self._batch_target_key].to(output.device)
        batch_targets = batch_targets.view(batch_targets.shape[0] * batch_targets.shape[1], -1).squeeze(-1)

        output = output[mask]
        batch_targets = batch_targets[mask]

        return self._loss(output, batch_targets)
