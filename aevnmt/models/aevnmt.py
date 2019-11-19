import numpy as np
from typing import Dict
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Independent, Categorical, Bernoulli

from aevnmt.components import RNNEncoder, tile_rnn_hidden, tile_rnn_hidden_for_decoder
from aevnmt.components.nli import DecomposableAttentionEncoder as NLIEncoder
from aevnmt.dist import get_named_params
from aevnmt.dist import NormalLayer, KumaraswamyLayer

from .generative import GenerativeLM, GenerativeTM
from .inference import InferenceEncoder, InferenceNetwork

from probabll.distributions import MixtureOfGaussians


class AEVNMT(nn.Module):

    def __init__(self, latent_size, src_embedder, tgt_embedder, encoder, decoder, 
            language_model: GenerativeLM, inf_encoder: InferenceEncoder,
            dropout, tied_embeddings, prior_family: str, prior_params: list, posterior_family: str,
            feed_z=False,  
            aux_lms: Dict[str, GenerativeLM]=dict(), aux_tms: Dict[str, GenerativeTM]=dict(),
            mixture_likelihood=False, mixture_likelihood_dir_prior=0.0):
        super().__init__()
        self.src_embedder = src_embedder
        self.tgt_embedder = tgt_embedder
        self.latent_size = latent_size
        self.language_model = language_model
        
        # TODO: use new GenerativeTM abstraction
        # v
        self.encoder = encoder  
        self.decoder = decoder
        self.feed_z = feed_z
        self.tied_embeddings = tied_embeddings
        if not tied_embeddings:
            self.output_matrix = nn.Parameter(torch.randn(tgt_embedder.num_embeddings, decoder.hidden_size))
        self.dropout_layer = nn.Dropout(p=dropout)
        self.encoder_init_layer = nn.Sequential(nn.Linear(latent_size, encoder.hidden_size),
                                                nn.Tanh())
        self.decoder_init_layer = nn.Sequential(nn.Linear(latent_size, decoder.hidden_size),
                                                nn.Tanh())
        # ^

        self.inf_network = InferenceNetwork(
            family=posterior_family,
            latent_size=latent_size,
            hidden_size=inf_encoder.hidden_size,
            encoder=inf_encoder)

        self.mixture_likelihood = mixture_likelihood
        self.mixture_likelihood_dir_prior = mixture_likelihood_dir_prior
        # Auxiliary LMs and TMs
        self.aux_lms = nn.ModuleDict(aux_lms)
        self.aux_tms = nn.ModuleDict(aux_tms)

        # This is done because the location and scale of the prior distribution are not considered
        # parameters, but are rather constant. Registering them as buffers still makes sure that
        # they will be moved to the appropriate device on which the model is run.
        self.prior_family = prior_family
        self.posterior_family = posterior_family
        if prior_family == "gaussian":
            if not prior_params:
                prior_params = [0., 1.]
            if len(prior_params) != 2:
                raise ValueError("Specify the Gaussian prior using a location and a strictly positive scale.")
            if prior_params[1] <= 0:
                raise ValueError("The scale of a Gaussian distribution is strictly positive: %r" % prior_params)
            self.register_buffer("prior_loc", torch.zeros([latent_size]))
            self.register_buffer("prior_scale", torch.ones([latent_size]))
        elif prior_family == "beta":
            if not prior_params:
                prior_params = [0.5, 0.5]
            if len(prior_params) != 2:
                raise ValueError("Specify the Beta prior using two strictly positive shape parameters.")
            if prior_params[0] <= 0. or prior_params[1] <= 0.:
                raise ValueError("The shape parameters of a Beta distribution are strictly positive: %r" % prior_params)
            self.register_buffer("prior_a", torch.full([latent_size], prior_params[0]))
            self.register_buffer("prior_b", torch.full([latent_size], prior_params[1]))
            if posterior_family != "kumaraswamy":
                raise ValueError("I think you forgot to change your posterior distribution to something with support (0,1)")
        elif prior_family == "mog":
            if not prior_params:
                prior_params = [10, 10, 0.5]
            if len(prior_params) != 3:
                raise ValueError("Specify the MoG prior using a number of components, a radius (for initialisation), and a strictly positive scale.")
            num_components = prior_params[0]
            if num_components <= 1:
                raise ValueError("An MoG prior requires more than 1 component.")
            prior_radius = prior_params[1]
            if prior_radius <= 0:
                raise ValueError("Initialising the MoG prior takes a strictly positive radius.")
            prior_scale = prior_params[2]
            if prior_scale <= 0:
                raise ValueError("The prior variance must be strictly positive.")
            # uniform prior over components
            self.register_buffer("prior_logits", torch.ones(num_components))
            self.register_buffer("prior_locations", - prior_radius + torch.rand([num_components, latent_size]) * 2 * prior_radius )
            self.register_buffer("prior_scales", torch.full([num_components, latent_size], prior_scale))
        else:
            raise NotImplementedError("I cannot impose a %s prior on the latent code." % prior_family)

    def inference_parameters(self):
        return self.inf_network.parameters()

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
        params = chain(self.encoder.parameters(),
                     self.decoder.parameters(),
                     self.tgt_embedder.parameters(),
                     self.encoder_init_layer.parameters(),
                     self.decoder_init_layer.parameters())
        if not self.tied_embeddings:
            params = chain(params, [self.output_matrix])
        return params

    def approximate_posterior(self, x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y):
        """
        Returns an approximate posterior distribution q(z|x, y).
        """
        return self.inf_network(x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y)

    def prior(self) -> torch.distributions.Distribution:
        if self.prior_family == "gaussian":
            p = Independent(torch.distributions.Normal(loc=self.prior_loc, scale=self.prior_scale), 1)
        elif self.prior_family == "beta":
            p = Independent(torch.distributions.Beta(self.prior_a, self.prior_b), 1)
        elif self.prior_family == "mog":
            p = MixtureOfGaussians(logits=self.prior_logits, locations=self.prior_locations, scales=self.prior_scales)
        return p

    def src_embed(self, x):
        # We share the source embeddings with the language_model.
        x_embed = self.language_model.embedder(x)
        x_embed = self.dropout_layer(x_embed)
        return x_embed

    def tgt_embed(self, y):
        y_embed = self.tgt_embedder(y)
        y_embed = self.dropout_layer(y_embed)
        return y_embed

    def encode(self, x, seq_len_x, z):
        x_embed = self.src_embed(x)
        hidden = tile_rnn_hidden(self.encoder_init_layer(z), self.encoder.rnn)
        return self.encoder(x_embed, seq_len_x, hidden=hidden)

    def init_decoder(self, encoder_outputs, encoder_final, z):
        self.decoder.init_decoder(encoder_outputs, encoder_final)
        hidden = tile_rnn_hidden_for_decoder(self.decoder_init_layer(z), self.decoder)
        return hidden

    def generate(self, pre_output):
        W = self.tgt_embedder.weight if self.tied_embeddings else self.output_matrix
        return F.linear(pre_output, W)

    def forward(self, x, seq_mask_x, seq_len_x, y, z):

        # Encode the source sentence and initialize the decoder hidden state.
        encoder_outputs, encoder_final = self.encode(x, seq_len_x, z)
        hidden = self.init_decoder(encoder_outputs, encoder_final, z)

        # Estimate the Categorical parameters for E[P(x|z)] using the given sample of the latent
        # variable.
        lm_likelihood = self.language_model(x, z)

        # Obtain auxiliary X_i|z, x_{<i}
        aux_lm_likelihoods = dict()
        for aux_name, aux_decoder in self.aux_lms.items():
            # TODO: give special treatment to components that take shuffled inputs, e.g. if aux_decoder.shuffled_inputs: aux_lm_likelihoods[aux_name] = aux_decoder(x_shuff, z)
            aux_lm_likelihoods[aux_name] = aux_decoder(x, z)

        # Obtain auxiliary Y_j|z, x, y_{<j}
        aux_tm_likelihoods = dict()
        for aux_name, aux_decoder in self.aux_tms.items():
            # TODO: give special treatment to components that take shuffled inputs, e.g. if aux_decoder.shuffled_inputs: aux_tm_likelihoods[aux_name] = aux_decoder(x, seq_mask_x, seq_len_x, y_shuff, z)
            aux_tm_likelihoods[aux_name] = aux_decoder(x, seq_mask_x, seq_len_x, y, z)

        # Estimate the Categorical parameters for E[P(y|x, z)] using the given sample of the latent
        # variable.
        tm_logits = []
        all_att_weights = []
        max_time = y.size(1)
        for t in range(max_time):
            prev_y = y[:, t]
            y_embed = self.tgt_embed(prev_y)
            pre_output, hidden, att_weights = self.decoder.step(y_embed, hidden, seq_mask_x,
                                                                encoder_outputs,z=z if self.feed_z else None)
            logits = self.generate(pre_output)
            tm_logits.append(logits)
            all_att_weights.append(att_weights)

        return torch.cat(tm_logits, dim=1), lm_likelihood, torch.cat(all_att_weights, dim=1), aux_lm_likelihoods, aux_tm_likelihoods

    def log_likelihood_tm(self, comp_name, likelihood: Distribution, y):
        # TODO: give special treatment to components that take shuffled inputs, e.g. if aux_decoder.shuffled_inputs: aux_lm_likelihoods[aux_name] = aux_decoder(x_shuff, z)
        return self.aux_tms[comp_name].log_prob(likelihood, y)
    
    def log_likelihood_lm(self, comp_name, likelihood: Distribution, x):
        # TODO: give special treatment to components that take shuffled inputs, e.g. if aux_decoder.shuffled_inputs: aux_tm_likelihoods[aux_name] = aux_decoder(x, seq_mask_x, seq_len_x, y_shuff, z)
        return self.aux_lms[comp_name].log_prob(likelihood, x)

    def compute_conditionals(self, x_in, seq_mask_x, seq_len_x, x_out, y_in, y_out, z):
        """
        :param x_in: [batch_size, max_length]
        :param seq_mask_x: [batch_size, max_length]
        :param seq_len_x: [batch_size]
        :param x_out: [batch_size, max_length]
        :param y_in: [batch_size, max_length]
        :param y_out: [batch_size, max_length]
        :param z: [batch_size, latent_size]
        :return: log p(x|z), log p(y|z,x)
        """
        # Encode the source sentence and initialize the decoder hidden state.
        encoder_outputs, encoder_final = self.encode(x_in, seq_len_x, z)
        hidden = self.init_decoder(encoder_outputs, encoder_final, z)

        # [B]
        lm_log_likelihood = self.language_model.log_prob(self.language_model(x_in, z), x_out)

        # Estimate the Categorical parameters for E[P(y|x, z)] using the given sample of the latent
        # variable.
        tm_logits = []
        max_time = y_in.size(1)
        for t in range(max_time):
            prev_y = y_in[:, t]
            y_embed = self.tgt_embed(prev_y)
            pre_output, hidden, _ = self.decoder.step(y_embed, hidden, seq_mask_x,
                                                                encoder_outputs,z=z if self.feed_z else None)
            logits = self.generate(pre_output)
            tm_logits.append(logits)
        # [max_length, batch_size, vocab_size]
        tm_logits = torch.cat(tm_logits, dim=1)
        # [batch_size, max_length, vocab_size]
        tm_logits = tm_logits.permute(0, 2, 1)
        # [batch_size]
        tm_loss = F.cross_entropy(tm_logits, y_out, ignore_index=self.tgt_embedder.padding_idx, reduction="none").sum(dim=1)

        return lm_log_likelihood, -tm_loss

    def compute_lm_likelihood(self, x_in, seq_mask_x, seq_len_x, x_out, z):
        """
        :param x_in: [batch_size, max_length]
        :param seq_mask_x: [batch_size, max_length]
        :param seq_len_x: [batch_size]
        :param x_out: [batch_size, max_length]
        :param y_in: [batch_size, max_length]
        :param y_out: [batch_size, max_length]
        :param z: [batch_size, latent_size]
        :return: log p(x|z)
        """
        # [B]
        return self.language_model.log_prob(self.language_model(x_in, z), x_out)

    def compute_tm_likelihood(self, x_in, seq_mask_x, seq_len_x, y_in, y_out, z):
        """
        :param x_in: [batch_size, max_length]
        :param seq_mask_x: [batch_size, max_length]
        :param seq_len_x: [batch_size]
        :param x_out: [batch_size, max_length]
        :param y_in: [batch_size, max_length]
        :param y_out: [batch_size, max_length]
        :param z: [batch_size, latent_size]
        :return: log p(x|z), log p(y|z,x)
        """
        # Encode the source sentence and initialize the decoder hidden state.
        encoder_outputs, encoder_final = self.encode(x_in, seq_len_x, z)
        hidden = self.init_decoder(encoder_outputs, encoder_final, z)

        # Estimate the Categorical parameters for E[P(y|x, z)] using the given sample of the latent
        # variable.
        tm_logits = []
        max_time = y_in.size(1)
        for t in range(max_time):
            prev_y = y_in[:, t]
            y_embed = self.tgt_embed(prev_y)
            pre_output, hidden, _ = self.decoder.step(y_embed, hidden, seq_mask_x,
                                                                encoder_outputs)
            logits = self.generate(pre_output)
            tm_logits.append(logits)
        # [max_length, batch_size, vocab_size]
        tm_logits = torch.cat(tm_logits, dim=1)

        # [batch_size, max_length, vocab_size]
        tm_logits = tm_logits.permute(0, 2, 1)

        # [batch_size]
        tm_loss = F.cross_entropy(tm_logits, y_out, ignore_index=self.tgt_embedder.padding_idx, reduction="none").sum(dim=1)

        return -tm_loss

    def loss(self, tm_logits, lm_likelihood: Categorical, targets_y, targets_x, qz, free_nats=0.,
            KL_weight=1., reduction="mean", aux_lm_likelihoods=dict(), aux_tm_likelihoods=dict()):
        """
        Computes an estimate of the negative evidence lower bound for the single sample of the latent
        variable that was used to compute the categorical parameters, and the distributions qz
        that the sample comes from.

        :param tm_logits: translation model logits, the unnormalized translation probabilities [B, T_y, vocab_size]
        :param lm_logits: language model logits, the unnormalized language probabilities [B, T_x, vocab_size]
        :param targets_y: target labels target sentence [B, T_y]
        :param targets_x: target labels source sentence [B, T_x]
        :param qz: distribution that was used to sample the latent variable.
        :param free_nats: KL = min(free_nats, KL)
        :param KL_weight: weight to multiply the KL with, applied after free_nats
        :param reduction: what reduction to apply, none ([B]), mean ([]) or sum ([])
        :param bow_logits: [B,src_vocab_size]
        :param bow_logits_tl: [B,tgt_vocab_size]
        :param ibm1_marginals: [B, T_y, tgt_vocab_size]
        """

        # Compute the loss for each batch element. Logits are of the form [B, T, vocab_size],
        # whereas the cross-entropy function wants a loss of the form [B, vocab_svocab_sizee, T].
        tm_logits = tm_logits.permute(0, 2, 1)
        tm_loss = F.cross_entropy(tm_logits, targets_y, ignore_index=self.tgt_embedder.padding_idx, reduction="none")
        tm_loss = tm_loss.sum(dim=1)
        tm_log_likelihood = -tm_loss

        # Compute the language model categorical loss.
        #lm_loss = self.language_model.loss(lm_logits, targets_x, reduction="none")
        lm_log_likelihood = self.language_model.log_prob(lm_likelihood, targets_x)
        lm_loss = - lm_log_likelihood

        # Compute the KL divergence between the distribution used to sample z, and the prior
        # distribution.
        pz = self.prior()  #.expand(qz.mean.size())

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
            # TODO: give special treatment to components that take shuffled outputs, e.g. if aux_decoder.shuffled_inputs: self.log_likelihood_lm(aux_name, aux_likelihood, targets_x_shuff) 
            side_lm_likelihood[c] = out_dict['lm/' + aux_name]
        # Alternative views of p(y|z,x)
        # [Cy, B]
        side_tm_likelihood = torch.zeros([len(aux_tm_likelihoods), KL.size(0)], dtype=KL.dtype, device=KL.device)
        for c, (aux_name, aux_likelihood) in enumerate(aux_tm_likelihoods.items()):
            out_dict['tm/' + aux_name] = self.log_likelihood_tm(aux_name, aux_likelihood, targets_y)
            # TODO: give special treatment to components that take shuffled outputs, e.g. if aux_decoder.shuffled_inputs: self.log_likelihood_lm(aux_name, aux_likelihood, targets_y_shuff) 
            side_tm_likelihood[c] = out_dict['tm/' + aux_name]
       
        if not self.mixture_likelihood:
            # ELBO
            #  E_q[ \log P(x|z,c=main) P(y|z,x,c=main)] - KL(q(z) || p(z)) 
            #  + E_q[\sum_{c not main} log P(x|z,c) + log P(y|z,x,c) ]
            # where the second row are heuristic side losses (we can think of it as multitask learning)
            elbo = tm_log_likelihood + lm_log_likelihood - KL
            # we sum the alternative views (as in multitask learning)
            aux_log_likelihood = side_lm_likelihood.sum(0) + side_tm_likelihood.sum(0)
            loss = - (elbo + aux_log_likelihood)
            # main log-likelihoods
            out_dict['lm/main'] = lm_log_likelihood
            out_dict['tm/main'] = tm_log_likelihood
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
