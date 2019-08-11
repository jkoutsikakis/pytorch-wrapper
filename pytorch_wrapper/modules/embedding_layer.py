import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    """
    Embedding Layer.
    """

    def __init__(self, vocab_size, emb_size, trainable, padding_idx=None):
        """
        :param vocab_size: Size of the vocabulary.
        :param emb_size: Size of the embeddings.
        :param trainable: Whether the embeddings should be altered during training.
        :param padding_idx: Index of the vector to be initialized with zeros.
        """
        super(EmbeddingLayer, self).__init__()

        self._embedding_layer = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self._embedding_layer.weight.requires_grad = trainable
        self._trainable = trainable

    def load_embeddings(self, embeddings):
        """
        Loads pre-trained embeddings.

        :param embeddings: Numpy array of the appropriate size containing the pre-trained embeddings.
        """
        self._embedding_layer.weight.data.copy_(torch.from_numpy(embeddings))

    def forward(self, x):
        return self._embedding_layer(x)
