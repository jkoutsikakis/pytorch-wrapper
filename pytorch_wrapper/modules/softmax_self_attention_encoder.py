import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import functional as pwF


class SoftmaxSelfAttentionEncoder(nn.Module):
    """
    Encodes a sequence using soft-max self-attention.
    """

    def __init__(self, attention_mlp, is_end_padded=True):
        """
        :param attention_mlp: MLP object used to generate unnormalized attention score(s). If the last dimension of the
            tensor returned by the MLP is larger than 1 then multi-attention is applied.
        :param is_end_padded: Whether to mask at the end.
        """

        super(SoftmaxSelfAttentionEncoder, self).__init__()
        self._attention_mlp = attention_mlp
        self._is_end_padded = is_end_padded

    def forward(self, batch_sequences, batch_sequence_lengths):
        """
        :param batch_sequences: 3D Tensor (batch_size, sequence_length, time_step_size).
        :param batch_sequence_lengths: 1D Tensor (batch_size) containing the lengths of the sequences.
        :return: Dict with a 2D Tensor (batch_size, time_step_size) or a 3D Tensor in case of multi-attention
            (batch_size, nb_attentions, time_step_size) containing the encodings (key=`output`) and a 2D Tensor
            (batch_size, sequence_length) or a 3D Tensor (batch_size, sequence_length, nb_attentions) containing the
            attention scores (key=`att_scores`).
        """

        att_scores = self._attention_mlp(batch_sequences)
        mask = pwF.create_mask_from_length(batch_sequence_lengths, batch_sequences.size(1),
                                           self._is_end_padded).unsqueeze(-1)

        masked_att_scores = att_scores.masked_fill(mask == 0, -1e9)

        masked_att_scores = F.softmax(masked_att_scores, dim=-2)
        masked_att_scores_t = torch.transpose(masked_att_scores, 1, 2)

        return {'output': torch.matmul(masked_att_scores_t, batch_sequences).squeeze(1),
                'att_scores': masked_att_scores.squeeze(2)}
