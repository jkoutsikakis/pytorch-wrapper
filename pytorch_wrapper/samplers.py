from torch.utils.data.sampler import Sampler


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
