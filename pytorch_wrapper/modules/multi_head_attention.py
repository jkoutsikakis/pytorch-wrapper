import torch.nn as nn
import torch.nn.functional as F
import torch
import math

from .. import functional as pwF


class MultiHeadAttention(nn.Module):
    """
    Multi Head Attention (https://arxiv.org/pdf/1706.03762.pdf).
    """

    def __init__(
            self,
            q_time_step_size,
            k_time_step_size,
            v_time_step_size,
            heads,
            attention_type='dot',
            dp=0,
            is_end_padded=True
    ):
        """
        :param q_time_step_size: Query time step size.
        :param k_time_step_size: Key time step size.
        :param v_time_step_size: Value time step size.
        :param heads: Number of attention heads.
        :param attention_type: Attention type ['dot', 'multiplicative', 'additive'].
        :param dp: Dropout probability.
        :param is_end_padded: Whether to mask at the end.
        """

        super(MultiHeadAttention, self).__init__()

        assert q_time_step_size % heads == 0, 'the number of heads must exactly divide the q_time_step_size'
        assert k_time_step_size % heads == 0, 'the number of heads must exactly divide the k_time_step_size'
        assert v_time_step_size % heads == 0, 'the number of heads must exactly divide the v_time_step_size'

        assert attention_type in ('dot', 'multiplicative', 'additive'), \
            'attention type must be dot, multiplicative, or additive'

        if attention_type == 'dot':
            assert q_time_step_size == k_time_step_size, \
                'in case of dot attention type q_time_step_size and k_time_step_size must be equal'

        self._q_d_k = q_time_step_size // heads
        self._k_d_k = k_time_step_size // heads
        self._v_d_k = v_time_step_size // heads
        self._h = heads
        self._v_time_step_size = v_time_step_size
        self._attention_type = attention_type

        self._dp = nn.Dropout(dp) if dp > 0 else None

        self._is_end_padded = is_end_padded
        if self._attention_type == 'multiplicative':
            multiplicative_w = torch.empty((1, heads, 1, self._k_d_k, self._q_d_k))
            nn.init.xavier_uniform_(multiplicative_w, gain=1.0)
            self._multiplicative_w = torch.nn.Parameter(multiplicative_w, requires_grad=True)

        elif self._attention_type == 'additive':

            bound = 1 / math.sqrt(self._q_d_k + self._k_d_k)
            additive_w1 = torch.empty((1, heads, 1, 1, self._q_d_k + self._k_d_k, self._q_d_k + self._k_d_k))
            nn.init.uniform_(additive_w1, -bound, bound)
            self._additive_w1 = torch.nn.Parameter(additive_w1, requires_grad=True)
            additive_b1 = torch.empty((1, heads, 1, 1, 1, self._q_d_k + self._k_d_k))
            nn.init.uniform_(additive_b1, -bound, bound)
            self._additive_b1 = torch.nn.Parameter(additive_b1, requires_grad=True)

            bound = 1 / math.sqrt(1)
            additive_w2 = torch.empty((1, heads, 1, 1, self._q_d_k + self._k_d_k, 1))
            nn.init.uniform_(additive_w2, -bound, bound)
            self._additive_w2 = torch.nn.Parameter(additive_w2, requires_grad=True)

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

        q = q.view(bs, q_len, self._h, self._q_d_k)
        k = k.view(bs, k_len, self._h, self._k_d_k)
        v = v.view(bs, v_len, self._h, self._v_d_k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self._attention_type == 'multiplicative':
            scores = torch.matmul(k.unsqueeze(3), self._multiplicative_w).squeeze(3)
            scores = torch.matmul(q, scores.transpose(2, 3)) / math.sqrt(self._q_d_k * self._k_d_k)
        elif self._attention_type == 'additive':
            q = q.unsqueeze(3).expand(bs, self._h, q_len, k_len, self._q_d_k)
            k = k.unsqueeze(2).expand(bs, self._h, q_len, k_len, self._k_d_k)
            scores = torch.cat([q, k], dim=-1).unsqueeze(4)
            scores = torch.tanh(torch.matmul(scores, self._additive_w1) + self._additive_b1)
            scores = torch.matmul(scores, self._additive_w2).squeeze(5).squeeze(4)
        else:
            scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self._q_d_k)
        mask = pwF.create_mask_from_length(k_sequence_lengths, k_len, self._is_end_padded).unsqueeze(1).unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

        scores = F.softmax(scores, dim=3)

        if self._dp is not None:
            scores_dp = self._dp(scores)
        else:
            scores_dp = scores

        attended = torch.matmul(scores_dp, v).transpose(1, 2)
        mask = pwF.create_mask_from_length(q_sequence_lengths, q_len, self._is_end_padded).unsqueeze(-1).unsqueeze(-1)
        attended = attended.masked_fill(mask == 0, 0)

        return {
            'output': attended.view(bs, q_len, self._v_time_step_size),
            'att_scores': scores
        }
