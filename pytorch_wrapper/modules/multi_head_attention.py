import torch.nn as nn
import torch.nn.functional as F
import torch
import math

from .. import functional as pwF


class MultiHeadAttention(nn.Module):
    """
    Multi Head Attention (https://arxiv.org/pdf/1706.03762.pdf).
    """

    def __init__(self, time_step_size, heads, dp=0, is_end_padded=True):
        """
        :param time_step_size: Time step size.
        :param heads: Number of attention heads.
        :param dp: Dropout probability.
        :param is_end_padded: Whether to mask at the end.
        """

        super(MultiHeadAttention, self).__init__()

        assert time_step_size % heads == 0, 'the number of heads must exactly divide the time_step_size'

        self._d_model = time_step_size
        self._d_k = time_step_size // heads
        self._h = heads
        self._time_step_size = time_step_size

        self._dp = nn.Dropout(dp) if dp > 0 else None

        self._q_linear = nn.Linear(time_step_size, time_step_size)
        self._v_linear = nn.Linear(time_step_size, time_step_size)
        self._k_linear = nn.Linear(time_step_size, time_step_size)
        self._out = nn.Linear(time_step_size, time_step_size)
        self._is_end_padded = is_end_padded

    def forward(self, q, k, v, q_sequence_lengths, k_sequence_lengths):
        """
        :param q: 3D Tensor (batch_size, q_sequence_length, time_step_size) containing the queries.
        :param k: 3D Tensor (batch_size, k_sequence_length, time_step_size) containing the keys.
        :param v: 3D Tensor (batch_size, k_sequence_length, time_step_size) containing the values.
        :param q_sequence_lengths: 1D Tensor (batch_size) containing the lengths of the query sequences.
        :param k_sequence_lengths: 1D Tensor (batch_size) containing the lengths of the key sequences.
        :return: 3D Tensor (batch_size, q_sequence_length, time_step_size).
        """

        bs = q.size(0)
        q_len = q.size(1)
        k_len = k.size(1)
        v_len = v.size(1)

        q = self._q_linear(q).view(bs, q_len, self._h, self._d_k)
        k = self._k_linear(k).view(bs, k_len, self._h, self._d_k)
        v = self._v_linear(v).view(bs, v_len, self._h, self._d_k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self._d_k)
        mask = pwF.create_mask_from_length(k_sequence_lengths, k_len, self._is_end_padded).unsqueeze(1).unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

        scores = F.softmax(scores, dim=3)

        if self._dp is not None:
            scores = self._dp(scores)

        attended = torch.matmul(scores, v).transpose(1, 2)
        mask = pwF.create_mask_from_length(q_sequence_lengths, q_len, self._is_end_padded).unsqueeze(-1).unsqueeze(-1)
        attended = attended.masked_fill(mask == 0, 0)

        concat = attended.view(bs, q_len, self._time_step_size)

        output = self._out(concat)

        return output
