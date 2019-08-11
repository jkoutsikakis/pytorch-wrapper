import torch.nn as nn


class Residual(nn.Module):
    """
    Adds the input of a module to it's output.
    """

    def __init__(self,
                 module,
                 residual_index=None,
                 model_output_key=None):
        """
        :param module: The module to wrap.
        :param residual_index: The index of the input to be added. Leave None if it is not a multi-input module.
        :param model_output_key: The key of the output of the model to be added. Leave None if it is not a multi-output
            module.
        """

        super(Residual, self).__init__()

        self._module = module
        self._residual_index = residual_index
        self._model_output_key = model_output_key

    def forward(self, *x):
        """
        :param x: The input of the wrapped module.
        :return: The output of the wrapped module added to it's input.
        """

        if self._residual_index is None:
            residual = x[0]
        else:
            residual = x[self._residual_index]

        out = self._module(*x)
        if self._model_output_key is None:
            out = out + residual
        else:
            out[self._model_output_key] = out[self._model_output_key] + residual

        return out
