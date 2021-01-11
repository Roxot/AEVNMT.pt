import numpy as np
from typing import Dict
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Categorical

from aevnmt.dist import PriorLayer, get_named_params
from aevnmt.components import label_smoothing_loss, Constraint, mmd_loss

from .generative import GenerativeLM, GenerativeTM
from .inference import InferenceModel


class SenVAE(nn.Module):

    def __init__(self, latent_size, src_embedder,
            language_model: GenerativeLM, inference_model: InferenceModel,
            prior: PriorLayer,
            dropout, tied_embeddings,
            feed_z=False,
            constraints=dict(),
            aux_lms: Dict[str, GenerativeLM]=dict(),
            mixture_likelihood=False, mixture_likelihood_dir_prior=0.0):
        super().__init__()
        self.src_embedder = src_embedder
        self.latent_size = latent_size
        self.language_model = language_model
        self.inference_model = inference_model

        self.mixture_likelihood = mixture_likelihood
        self.mixture_likelihood_dir_prior = mixture_likelihood_dir_prior

        # Auxiliary LMs and TMs
        self.aux_lms = nn.ModuleDict(aux_lms)

        self.constraints = nn.ModuleDict(constraints)

        # This is done because the location and scale of the prior distribution are not considered
        # parameters, but are rather constant. Registering them as buffers still makes sure that
        # they will be moved to the appropriate device on which the model is run.
        self.prior = prior

    def inference_parameters(self):
        return self.inference_model.parameters()

    def embedding_parameters(self):
        return chain(self.src_embedder.parameters())

    def generative_parameters(self):
        return chain(self.lm_parameters(), self.aux_lm_parameters())

    def aux_lm_parameters(self):
        return chain(*[model.parameters() for model in self.aux_lms.values()])

    def lm_parameters(self):
        return chain(self.src_embedder.parameters(), self.language_model.parameters())

    def lagrangian_parameters(self):
        if self.constraints:
            return self.constraints.parameters()
        return None

    def approximate_posterior(self, x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y):
        """
        Returns an approximate posterior distribution q(z|x, y).
        """
        return self.inference_model(x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y)

    def src_embed(self, x):
        x_embed = self.src_embedder(x)
        x_embed = self.dropout_layer(x_embed)
        return x_embed

    def forward(self, x, seq_mask_x, seq_len_x, z):
        """
        Run all components of the model and return parameterised distributions.

        Returns:
            tm_likelihood: Categorical distributions Y_j|x,z, y_{<j} with shape [B, Ty, Vy]
            lm_likelihood: Categorical distributions X_i|z,x_{<i} with shape [B, Tx, Vx]
            state: a dictionary with information from decoders' forward passes (e.g. `att_weights`)
            aux_lm_likelihoods: dictionary mapping auxiliary LMs to their parameterised distributions
            aux_tm_likelihoods: dictionary mapping auxiliary TMs to their parameterised distributions
        """
        state = dict()
        lm_likelihood = self.language_model(x, z, state)

        # Obtain auxiliary X_i|z, x_{<i}
        aux_lm_likelihoods = dict()
        for aux_name, aux_decoder in self.aux_lms.items():
            aux_lm_likelihoods[aux_name] = aux_decoder(x, z)

        return lm_likelihood, state, aux_lm_likelihoods

    def log_likelihood_lm(self, comp_name, likelihood: Distribution, x):
        return self.aux_lms[comp_name].log_prob(likelihood, x)
