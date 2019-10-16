import random

from torch.utils.data.sampler import Sampler


def _batchify(l, batch_size):
    for i in range(0, len(l), batch_size):
        yield l[i:i + batch_size]


def _flatten(l):
    return [item for sublist in l for item in sublist]


class SubsetSequentialSampler(Sampler):
    """
    Samples elements sequentially based on a list of indexes.
    """

    def __init__(self, indexes):
        """
        :param indexes: a list of indexes.
        """
        super(SubsetSequentialSampler, self).__init__(None)
        self._indexes = indexes

    def __iter__(self):
        return (self._indexes[i] for i in range(len(self._indexes)))

    def __len__(self):
        return len(self._indexes)


class OrderedBatchWiseRandomSampler(Sampler):
    """
    Semi-randomly samples indexes from a dataset ensuring that the corresponding examples will have similar values.
    Values are returned by a callable.
    """

    def __init__(self, data_source, get_order_value_callable, batch_size, seed=1234):
        """
        :param data_source: a data source (usually a dataset object).
        :param get_order_value_callable: a callable that takes as input the example's index and returns the ordering
            value.
        :param batch_size: the batch size.
        :param seed: the initial seed.
        """
        super(OrderedBatchWiseRandomSampler, self).__init__(None)
        self._sorted_indexes = sorted(list(range(len(data_source))), key=lambda x: get_order_value_callable(x))
        self._batch_size = batch_size
        self._current_seed = seed

    def __iter__(self):
        self._current_seed += 1
        rand_state = random.Random(self._current_seed)
        indexes = list(_batchify(self._sorted_indexes.copy(), self._batch_size))
        rand_state.shuffle(indexes)
        return iter(_flatten(indexes))

    def __len__(self):
        return len(self._sorted_indexes)


class SubsetOrderedBatchWiseRandomSampler(Sampler):
    """
    Semi-randomly samples indexes from a list ensuring that the corresponding examples will have similar values. Values
    are returned by a callable.
    """

    def __init__(self, indexes, get_order_value_callable, batch_size, seed=1234):
        """
        :param indexes: a list of indexes.
        :param get_order_value_callable: a callable that takes as input the example's index and returns the ordering
            value.
        :param batch_size: the batch size.
        :param seed: the initial seed.
        """
        super(SubsetOrderedBatchWiseRandomSampler, self).__init__(None)
        self._sorted_indexes = sorted(indexes, key=lambda i: get_order_value_callable(i))
        self._batch_size = batch_size
        self._current_seed = seed

    def __iter__(self):
        self._current_seed += 1
        rand_state = random.Random(self._current_seed)
        indexes = list(_batchify(self._sorted_indexes.copy(), self._batch_size))
        rand_state.shuffle(indexes)
        return iter(_flatten(indexes))

    def __len__(self):
        return len(self._sorted_indexes)


class OrderedSequentialSampler(Sampler):
    """
    Samples elements from a dataset ordered by a value returned by a callable for each example.
    """

    def __init__(self, data_source, get_order_value_callable):
        """
        :param data_source: a data source (usually a dataset object).
        :param get_order_value_callable: a callable that takes as input the example's index and returns the ordering
            value.
        """
        super(OrderedSequentialSampler, self).__init__(None)
        self._sorted_indexes = sorted(list(range(len(data_source))), key=lambda i: get_order_value_callable(i))

    def __iter__(self):
        return iter(self._sorted_indexes)

    def __len__(self):
        return len(self._sorted_indexes)


class SubsetOrderedSequentialSampler(Sampler):
    """
    Samples elements from a list of indexes ordered by a value returned by a callable for each example.
    """

    def __init__(self, indexes, get_order_value_callable):
        """
        :param indexes: a list of indexes.
        :param get_order_value_callable: a callable that takes as input the example's index and returns the ordering
            value.
        """
        super(SubsetOrderedSequentialSampler, self).__init__(None)
        self._sorted_indexes = sorted(indexes, key=lambda i: get_order_value_callable(i))

    def __iter__(self):
        return iter(self._sorted_indexes)

    def __len__(self):
        return len(self._sorted_indexes)
