import torch
import torch.nn as nn


class DetachedEmbeddingLayer(nn.Module):
    """
    Wrapper around nn.Embedding.forward(inputs) that returns detached outputs.
    Use this to share generative embeddings with inference models.
    """

    def __init__(self, embedder: nn.Embedding):
        super().__init__()
        self.embedder = embedder

    @property
    def num_embeddings(self):
        return self.embedder.num_embeddings
    
    @property
    def embedding_dim(self):
        return self.embedder.embedding_dim

    @property
    def padding_idx(self):
        return self.embedder.padding_idx

    def forward(self, inputs):
        return self.embedder(inputs).detach()

