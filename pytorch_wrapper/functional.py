import torch


def create_mask_from_length(length_tensor, mask_size, zeros_at_end=True):
    """
    Creates a binary mask based on length.

    :param length_tensor: ND Tensor containing the lengths.
    :param mask_size: Int specifying the mask size. Usually the largest length.
    :param zeros_at_end: Whether to put the zeros of the mask at the end.
    :return: (N+1)D Int Tensor (..., mask_size).
    """

    if zeros_at_end:
        mask = torch.arange(0, mask_size, dtype=torch.int, device=length_tensor.device)
    else:
        mask = torch.arange(mask_size - 1, -1, step=-1, dtype=torch.int, device=length_tensor.device)

    mask = mask.int().view([1] * (len(length_tensor.shape)) + [-1])

    return mask < length_tensor.int().unsqueeze(-1)


def masked_max_pooling(data_tensor, mask, dim):
    """
    Performs masked max-pooling across the specified dimension of a Tensor.

    :param data_tensor: ND Tensor.
    :param mask: Tensor containing a binary mask that can be broad-casted to the shape of data_tensor.
    :param dim: Int that corresponds to the dimension.
    :return: (N-1)D Tensor containing the result of the max-pooling operation.
    """

    if dim < 0:
        dim = len(data_tensor.shape) + dim

    mask = mask.view(list(mask.shape) + [1] * (len(data_tensor.shape) - len(mask.shape)))
    data_tensor = data_tensor.masked_fill(mask == 0, -1e9)

    max_vals, max_ids = torch.max(data_tensor, dim=dim)

    return max_vals


def masked_min_pooling(data_tensor, mask, dim):
    """
    Performs masked min-pooling across the specified dimension of a Tensor.

    :param data_tensor: ND Tensor.
    :param mask: Tensor containing a binary mask that can be broad-casted to the shape of data_tensor.
    :param dim: Int that corresponds to the dimension.
    :return: (N-1)D Tensor containing the result of the min-pooling operation.
    """

    if dim < 0:
        dim = len(data_tensor.shape) + dim

    mask = mask.view(list(mask.shape) + [1] * (len(data_tensor.shape) - len(mask.shape)))
    data_tensor = data_tensor.masked_fill(mask == 0, 1e9)

    min_vals, min_ids = torch.min(data_tensor, dim=dim)

    return min_vals


def masked_mean_pooling(data_tensor, mask, dim):
    """
    Performs masked mean-pooling across the specified dimension of a Tensor.

    :param data_tensor: ND Tensor.
    :param mask: Tensor containing a binary mask that can be broad-casted to the shape of data_tensor.
    :param dim: Int that corresponds to the dimension.
    :return: (N-1)D Tensor containing the result of the mean-pooling operation.
    """

    if dim < 0:
        dim = len(data_tensor.shape) + dim

    mask = mask.view(list(mask.shape) + [1] * (len(data_tensor.shape) - len(mask.shape)))
    data_tensor = data_tensor.masked_fill(mask == 0, 0)

    nominator = torch.sum(data_tensor, dim=dim)
    denominator = torch.sum(mask.type(nominator.type()), dim=dim)

    return nominator / denominator


def get_first_non_masked_element(data_tensor, lengths_tensor, dim, is_end_padded=True):
    """
    Returns the first non masked element of a Tensor along the specified dimension.

    :param data_tensor: ND Tensor.
    :param lengths_tensor: (dim)D Tensor containing lengths.
    :param dim: Int that corresponds to the dimension.
    :param is_end_padded: Whether the Tensor is padded at the end.
    :return: (N-1)D Tensor containing the first non-masked elements along the specified dimension.
    """

    if dim < 0:
        dim = len(data_tensor.shape) + dim

    if is_end_padded:
        idx = torch.tensor(0, dtype=torch.long, device=data_tensor.device)
        idx = idx.view([1] * len(data_tensor.shape))
        shape_to_expand = list(data_tensor.shape)
        shape_to_expand[dim] = 1
        idx = idx.expand(shape_to_expand)
    else:
        idx = (data_tensor.shape[dim] - lengths_tensor).long()
        idx = idx.view(list(idx.shape) + [1] * (len(data_tensor.shape) - len(idx.shape)))
        shape_to_expand = list(data_tensor.shape)
        shape_to_expand[dim] = 1
        idx = idx.expand(shape_to_expand)

    return data_tensor.gather(dim, idx).squeeze(dim=dim)


def get_last_non_masked_element(data_tensor, lengths_tensor, dim, is_end_padded=True):
    """
    Returns the last non masked element of a Tensor along the specified dimension.

    :param data_tensor: ND Tensor.
    :param lengths_tensor: (dim)D Tensor containing lengths.
    :param dim: Int that corresponds to the dimension.
    :param is_end_padded: Whether the Tensor is padded at the end.
    :return: (N-1)D Tensor containing the last non-masked elements along the specified dimension.
    """

    if dim < 0:
        dim = len(data_tensor.shape) + dim

    if is_end_padded:
        idx = (lengths_tensor - 1).long()
        idx = idx.view(list(idx.shape) + [1] * (len(data_tensor.shape) - len(idx.shape)))
        shape_to_expand = list(data_tensor.shape)
        shape_to_expand[dim] = 1
        idx = idx.expand(shape_to_expand)
    else:
        idx = torch.tensor(data_tensor.shape[dim] - 1, dtype=torch.long, device=data_tensor.device)
        idx = idx.view([1] * len(data_tensor.shape))
        shape_to_expand = list(data_tensor.shape)
        shape_to_expand[dim] = 1
        idx = idx.expand(shape_to_expand)

    return data_tensor.gather(dim, idx).squeeze(dim=dim)


def get_last_state_of_rnn(rnn_out, batch_sequence_lengths, is_bidirectional, is_end_padded=True):
    """
    Returns the last state(s) of the output of an RNN.

    :param rnn_out: 3D Tensor (batch_size, sequence_length, time_step_size).
    :param batch_sequence_lengths: 1D Tensor (batch_size) containing the lengths of the sequences.
    :param is_bidirectional: Whether the RNN is bidirectional or not.
    :param is_end_padded: Whether the Tensor is padded at the end.
    :return: 2D Tensor (batch_size, time_step_size or time_step_size * 2) containing the last state(s) of the RNN.
    """

    out = get_last_non_masked_element(rnn_out, lengths_tensor=batch_sequence_lengths, dim=1,
                                      is_end_padded=is_end_padded)

    if is_bidirectional:
        backward_out = get_first_non_masked_element(rnn_out, lengths_tensor=batch_sequence_lengths, dim=1,
                                                    is_end_padded=is_end_padded)

        out = torch.cat([out[:, :rnn_out.shape[2] // 2], backward_out[:, rnn_out.shape[2] // 2:]], dim=1)

    return out


def pad(data_tensor, pad_size, dim, pad_at_end=True):
    """
    Pads a Tensor with zeros along a dimension.

    :param data_tensor: Tensor to pad.
    :param pad_size: How many zeros to append.
    :param dim: The dimension to pad.
    :param pad_at_end: Whether to pad at the end.
    :return: Padded Tensor.
    """

    if dim < 0:
        dim = len(data_tensor.shape) + dim

    shape = list(data_tensor.shape)
    shape[dim] = pad_size

    zeros = torch.zeros(shape, device=data_tensor.device)

    if pad_at_end:
        return torch.cat([data_tensor, zeros], dim=dim)
    else:
        return torch.cat([zeros, data_tensor], dim=dim)


def same_dropout(data_tensor, dropout_p, dim, is_model_training):
    """
    Drops the same random elements of a Tensor across the specified dimension, during training.

    :param data_tensor: ND Tensor.
    :param dropout_p: The dropout probability.
    :param dim: Int that corresponds to the dimension.
    :param is_model_training: Whether the model is currently training.
    :return: ND Tensor.
    """

    if dim < 0:
        dim = len(data_tensor.shape) + dim

    if dropout_p is None or dropout_p == 0 or not is_model_training:
        return data_tensor

    assert 0 <= dropout_p < 1, 'dropout probability must be in range [0,1)'

    shape = list(data_tensor.shape)
    shape[dim] = 1
    dp = torch.empty(*shape, dtype=torch.float, device=data_tensor.device)
    dp = torch.bernoulli(dp.fill_((1 - dropout_p))) / (1 - dropout_p)

    return data_tensor * dp


def sub_tensor_dropout(data_tensor, dropout_p, dim, is_model_training):
    """
    Drops (zeroes-out) random sub-Tensors of a Tensor across the specified dimension, during training.

    :param data_tensor: ND Tensor.
    :param dropout_p: The dropout probability.
    :param dim: Int that corresponds to the dimension.
    :param is_model_training: Whether the model is currently training.
    :return: ND Tensor.
    """

    if dim < 0:
        dim = len(data_tensor.shape) + dim

    if dropout_p is None or dropout_p == 0 or not is_model_training:
        return data_tensor

    assert 0 <= dropout_p < 1, 'dropout probability must be in range [0,1)'

    dp = torch.empty(*(data_tensor.shape[:dim + 1]), dtype=torch.float, device=data_tensor.device)
    dp = torch.bernoulli(dp.fill_((1 - dropout_p)))
    dp = dp.view(list(dp.shape) + [1] * (len(data_tensor.shape) - len(dp.shape)))

    return data_tensor * dp
