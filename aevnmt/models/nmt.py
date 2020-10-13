import torch
import torch.nn as nn
import torch.nn.functional as F
from aevnmt.components import label_smoothing_loss
from torch.distributions import Categorical


class ConditionalNMT(nn.Module):

    def __init__(self, translation_model):
        super().__init__()
        self.translation_model = translation_model

    def generative_parameters(self):
        return self.translation_model.parameters()

    def inference_parameters(self):
        return None

    def lagrangian_parameters(self):
        return None

    def forward(self, x, seq_mask_x, seq_len_x, y) -> Categorical:
        state = dict()
        likelihood = self.translation_model(x, seq_mask_x, seq_len_x, y, None, state)
        return likelihood, state

    def loss(self, likelihood, targets, reduction, label_smoothing=0.):
        """
        Computes the negative categorical log-likelihood for the given model output.

        :param logits: outputs of the model, the unnormalized probabilities [B, T, vocab_size]
        :param targets: target labels [B, T]
        :param reduction: what reduction to apply, none ([B]), mean ([]) or sum ([])
        """

        log_likelihood = self.translation_model.log_prob(likelihood, targets)
        if label_smoothing > 0:
            smooth_loss = label_smoothing_loss(likelihood, targets,
                                               ignore_index=self.translation_model.tgt_embedder.padding_idx)
            log_likelihood = (1-label_smoothing) * log_likelihood + label_smoothing * smooth_loss
        loss = - log_likelihood

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        out_dict = {"loss": loss}
        return out_dict
