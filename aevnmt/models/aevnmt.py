import numpy as np
from typing import Dict
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Categorical 

from aevnmt.dist import PriorLayer, get_named_params
from aevnmt.components import label_smoothing_loss

from .generative import GenerativeLM, GenerativeTM
from .inference import InferenceModel


class AEVNMT(nn.Module):

    def __init__(self, latent_size, src_embedder, tgt_embedder, 
            language_model: GenerativeLM, translation_model: GenerativeTM, inference_model: InferenceModel,
            prior: PriorLayer,
            dropout, tied_embeddings,
            feed_z=False,
            mdr=-1.,
            aux_lms: Dict[str, GenerativeLM]=dict(), aux_tms: Dict[str, GenerativeTM]=dict(),
            mixture_likelihood=False, mixture_likelihood_dir_prior=0.0):
        super().__init__()
        self.src_embedder = src_embedder
        self.tgt_embedder = tgt_embedder
        self.latent_size = latent_size
        self.language_model = language_model
        self.translation_model = translation_model
        self.inference_model = inference_model

        self.mixture_likelihood = mixture_likelihood
        self.mixture_likelihood_dir_prior = mixture_likelihood_dir_prior
        # Auxiliary LMs and TMs
        self.aux_lms = nn.ModuleDict(aux_lms)
        self.aux_tms = nn.ModuleDict(aux_tms)

        self.mdr = mdr
        if mdr > 0.:
            self.mdr_lag_weight = torch.nn.Parameter(torch.tensor([1.]))

        # This is done because the location and scale of the prior distribution are not considered
        # parameters, but are rather constant. Registering them as buffers still makes sure that
        # they will be moved to the appropriate device on which the model is run.
        self.prior = prior

    def inference_parameters(self):
        return self.inference_model.parameters()

    def embedding_parameters(self):
        return chain(self.src_embedder.parameters(), self.tgt_embedder.parameters())

    def generative_parameters(self):
        return chain(self.lm_parameters(), self.tm_parameters(), self.aux_lm_parameters(), self.aux_tm_parameters())

    def aux_lm_parameters(self):
        return chain(*[model.parameters() for model in self.aux_lms.values()])
    
    def aux_tm_parameters(self):
        return chain(*[model.parameters() for model in self.aux_tms.values()])

    def lm_parameters(self):
        return chain(self.src_embedder.parameters(), self.language_model.parameters())  

    def tm_parameters(self):
        return chain(self.tgt_embedder.parameters(), self.translation_model.parameters())
   
    def lagrangian_parameters(self):
        if self.mdr > 0.:
            return [self.mdr_lag_weight]
        else:
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

    def tgt_embed(self, y):
        y_embed = self.tgt_embedder(y)
        y_embed = self.dropout_layer(y_embed)
        return y_embed

    def forward(self, x, seq_mask_x, seq_len_x, y, z):
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
        tm_likelihood = self.translation_model(x, seq_mask_x, seq_len_x, y, z, state)

        # Obtain auxiliary X_i|z, x_{<i}
        aux_lm_likelihoods = dict()
        for aux_name, aux_decoder in self.aux_lms.items():
            aux_lm_likelihoods[aux_name] = aux_decoder(x, z)

        # Obtain auxiliary Y_j|z, x, y_{<j}
        aux_tm_likelihoods = dict()
        for aux_name, aux_decoder in self.aux_tms.items():
            aux_tm_likelihoods[aux_name] = aux_decoder(x, seq_mask_x, seq_len_x, y, z)

        return tm_likelihood, lm_likelihood, state, aux_lm_likelihoods, aux_tm_likelihoods

    def log_likelihood_tm(self, comp_name, likelihood: Distribution, y):
        return self.aux_tms[comp_name].log_prob(likelihood, y)
    
    def log_likelihood_lm(self, comp_name, likelihood: Distribution, x):
        return self.aux_lms[comp_name].log_prob(likelihood, x)

    def loss(self, tm_likelihood: Categorical, lm_likelihood: Categorical, targets_y, targets_x, qz: Distribution, 
            free_nats=0., KL_weight=1., ls_y=0., ls_x=0., reduction="mean", aux_lm_likelihoods=dict(), 
            aux_tm_likelihoods=dict(), loss_cfg=set()):
        """
        Computes an estimate of the negative evidence lower bound for the single sample of the latent
        variable that was used to compute the categorical parameters, and the distributions qz
        that the sample comes from.

        :param tm_likelihood: Categorical distributions from LM with shape [B, Ty, Vy]
        :param lm_likelihood: Categorical distributions from TM with shape [B, Tx, Vx]
        :param targets_y: target labels target sentence [B, T_y]
        :param targets_x: target labels source sentence [B, T_x]
        :param qz: distribution that was used to sample the latent variable.
        :param free_nats: KL = min(free_nats, KL)
        :param KL_weight: weight to multiply the KL with, applied after free_nats
        :param reduction: what reduction to apply, none ([B]), mean ([]) or sum ([])
        :param aux_lm_likelihoods: a dictionary with LM likelihoods
        :param aux_tm_likelihoods: a dictionary with TM likelihoods
        """
        # [B]
        tm_log_likelihood = self.translation_model.log_prob(tm_likelihood, targets_y)
        if ls_y > 0:
            smoothing_y = label_smoothing_loss(tm_log_likelihood, targets_y, 
                                               ignore_index=self.translation_model.tgt_embedder.padding_idx)
            tm_log_likelihood = (1 - ls_y) * tm_log_likelihood + ls_y * smoothing_y
        tm_loss = - tm_log_likelihood

        # [B]
        lm_log_likelihood = self.language_model.log_prob(lm_likelihood, targets_x)
        if ls_x > 0:
            smoothing_x = label_smoothing_loss(lm_likelihood, targets_x,
                                               ignore_index=self.language_model.pad_idx)
            lm_log_likelihood = (1 - ls_x) * lm_log_likelihood + ls_x * smoothing_x
        lm_loss = - lm_log_likelihood

        # Compute the KL divergence between the distribution used to sample z, and the prior
        # distribution.
        pz = self.prior()
        KL = torch.distributions.kl.kl_divergence(qz, pz)
        raw_KL = KL * 1

        if free_nats > 0:
            KL = torch.clamp(KL, min=free_nats)
        KL *= KL_weight

        out_dict = dict()
        out_dict['KL'] = KL
        out_dict['raw_KL'] = raw_KL
       
        # Alternative views of p(x|z)
        # [Cx, B]
        side_lm_likelihood = torch.zeros([len(aux_lm_likelihoods), KL.size(0)], dtype=KL.dtype, device=KL.device)
        for c, (aux_name, aux_likelihood) in enumerate(aux_lm_likelihoods.items()):
            out_dict['lm/' + aux_name] = self.log_likelihood_lm(aux_name, aux_likelihood, targets_x)
            side_lm_likelihood[c] = out_dict['lm/' + aux_name]
        # Alternative views of p(y|z,x)
        # [Cy, B]
        side_tm_likelihood = torch.zeros([len(aux_tm_likelihoods), KL.size(0)], dtype=KL.dtype, device=KL.device)
        for c, (aux_name, aux_likelihood) in enumerate(aux_tm_likelihoods.items()):
            out_dict['tm/' + aux_name] = self.log_likelihood_tm(aux_name, aux_likelihood, targets_y)
            side_tm_likelihood[c] = out_dict['tm/' + aux_name]

        if not self.mixture_likelihood:
            # ELBO
            #  E_q[ \log P(x|z,c=main) P(y|z,x,c=main)] - KL(q(z) || p(z)) 
            #  + E_q[\sum_{c not main} log P(x|z,c) + log P(y|z,x,c) ]
            # where the second row are heuristic side losses (we can think of it as multitask learning)
            elbo = tm_log_likelihood + lm_log_likelihood - KL
            # we sum the alternative views (as in multitask learning)
            aux_log_likelihood = side_lm_likelihood.sum(0) + side_tm_likelihood.sum(0)

            # main log-likelihoods
            out_dict['lm/main'] = lm_log_likelihood
            out_dict['tm/main'] = tm_log_likelihood
            
            if not loss_cfg:
                loss = - (elbo + aux_log_likelihood)
            else:
                loss = KL
                for name in loss_cfg:
                    if name in out_dict:
                        loss = loss - out_dict[name]
        else: 
            # ELBO uses mixture models for X|z and Y|z,x:
            #  E_q[ \log P(x|z) + \log P(y|z,x)] - KL(q(z) || p(z))
            #   where \log P(x|z)   = \log \sum_{c=1}^{Cy} w_c P(x|z,c)
            #   and   \log P(y|z,x) = \log \sum_{c=1}^{Cx} w_c P(y|z,x,c)
            Cx = len(aux_lm_likelihoods) + 1
            if self.mixture_likelihood_dir_prior == 0:
                wx = torch.full([KL.size(0), Cx], 1. / Cx, dtype=KL.dtype, device=KL.device).permute(1, 0)
            else:
                wx = torch.distributions.Dirichlet(
                    torch.full([KL.size(0), Cx], self.mixture_likelihood_dir_prior, 
                        dtype=KL.dtype, device=KL.device)).sample().permute(1, 0)
            # [Cx, B] -> [B]
            lm_mixture = (torch.cat([lm_log_likelihood.unsqueeze(0), side_lm_likelihood]) - torch.log(wx)).logsumexp(0)
            Cy = len(aux_tm_likelihoods) + 1
            if self.mixture_likelihood_dir_prior == 0:
                wy = torch.full([KL.size(0), Cy], 1. / Cy, dtype=KL.dtype, device=KL.device).permute(1, 0)
            else:
                wy = torch.distributions.Dirichlet(
                    torch.full([KL.size(0), Cy], self.mixture_likelihood_dir_prior, 
                        dtype=KL.dtype, device=KL.device)).sample().permute(1, 0)
            # [Cy, B] -> [B]
            tm_mixture = (torch.cat([tm_log_likelihood.unsqueeze(0), side_tm_likelihood]) - torch.log(wy)).logsumexp(0)
            elbo = lm_mixture + tm_mixture - KL
            loss = - elbo
            out_dict['lm/main'] = lm_mixture
            out_dict['tm/main'] = tm_mixture
            out_dict['lm/recurrent'] = lm_log_likelihood
            out_dict['tm/recurrent'] = tm_log_likelihood

        # Add MDR constraint.
        if self.mdr > 0.:
            constraint = self.mdr_lag_weight * (self.mdr - KL.mean()) 
            loss = loss + constraint

        # Return differently according to the reduction setting.
        if reduction == "mean":
            out_dict['loss'] = loss.mean()
        elif reduction == "sum":
            out_dict['loss'] = loss.sum()
        elif reduction == "none":
            out_dict['loss'] = loss
        else:
            raise Exception(f"Unknown reduction option {reduction}")

        return out_dict
