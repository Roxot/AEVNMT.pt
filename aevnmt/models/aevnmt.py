import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent

from aevnmt.components import RNNEncoder, tile_rnn_hidden, tile_rnn_hidden_for_decoder
from aevnmt.components.nli import DecomposableAttentionEncoder as NLIEncoder
from aevnmt.dist import get_named_params
from aevnmt.dist import NormalLayer, KumaraswamyLayer
from probabll.distributions import MixtureOfGaussians
from .inference import get_inference_encoder
from .inference import InferenceNetwork

from itertools import chain


class AEVNMT(nn.Module):

    def __init__(self, tgt_vocab_size, emb_size, latent_size, encoder, decoder, language_model,
            pad_idx, dropout, tied_embeddings, prior_family: str, prior_params: list, posterior_family: str,
            inf_encoder_style: str, inf_conditioning: str, bow=False, bow_tl=False):

        super().__init__()
        self.latent_size = latent_size
        self.pad_idx = pad_idx
        self.encoder = encoder
        self.decoder = decoder
        self.language_model = language_model
        self.tgt_embedder = nn.Embedding(tgt_vocab_size, emb_size, padding_idx=pad_idx)
        self.tied_embeddings = tied_embeddings
        if not tied_embeddings:
            self.output_matrix = nn.Parameter(torch.randn(tgt_vocab_size, decoder.hidden_size))
        self.dropout_layer = nn.Dropout(p=dropout)
        self.encoder_init_layer = nn.Sequential(nn.Linear(latent_size, encoder.hidden_size),
                                                nn.Tanh())
        self.decoder_init_layer = nn.Sequential(nn.Linear(latent_size, decoder.hidden_size),
                                                nn.Tanh())
        self.lm_init_layer = nn.Sequential(nn.Linear(latent_size, language_model.hidden_size),
                                           nn.Tanh())

        self.inf_network = InferenceNetwork(
            family=posterior_family,
            latent_size=latent_size,
            # TODO: there's too much overloading of hyperparameters, why are we using the specs from the generative encoder???
            hidden_size=encoder.hidden_size,
            encoder=get_inference_encoder(
                encoder_style=inf_encoder_style,
                conditioning_context=inf_conditioning,
                embedder_x=self.language_model.embedder,
                embedder_y=self.tgt_embedder,
                hidden_size=encoder.hidden_size,
                rnn_bidirectional=encoder.bidirectional,
                rnn_num_layers=encoder.num_layers,
                rnn_cell_type=encoder.cell_type,
                transformer_heads=8,  # TODO: create a hyperparameter for this
                transformer_layers=8, # TODO: create a hyperparameter for this
                nli_shared_size=self.language_model.embedder.embedding_dim,
                nli_max_distance=20,  # TODO: create a hyperaparameter for this
                dropout=dropout))


        self.bow_output_layer=None
        self.bow_output_layer_tl=None

        if bow:
            self.bow_output_layer = nn.Linear(latent_size,
                                              self.language_model.embedder.num_embeddings, bias=True)

        if bow_tl:
            self.bow_output_layer_tl = nn.Linear(latent_size,
                                              self.tgt_embedder.num_embeddings, bias=True)

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

    def generative_parameters(self):
        # TODO: separate the generative model into a GenerativeModel module
        #  within that module, have two modules, namely, LanguageModel and TranslationModel
        return chain(self.lm_parameters(), self.tm_parameters(), self.bow_parameters())

    def bow_parameters(self):
        if self.bow_output_layer is None:
            bow_params=iter(())
        else:
            bow_params=self.bow_output_layer.parameters()

        if self.bow_output_layer_tl is None:
            bow_params_tl=iter(())
        else:
            bow_params_tl=self.bow_output_layer_tl.parameters()

        return chain( bow_params  , bow_params_tl   )

    def lm_parameters(self):
        return chain(self.language_model.parameters(), self.lm_init_layer.parameters())

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

    def run_language_model(self, x, z):
        """
        Runs the language_model.

        :param x: unembedded source sentence
        :param z: a sample of the latent variable
        """
        hidden = tile_rnn_hidden(self.lm_init_layer(z), self.language_model.rnn)
        return self.language_model(x, hidden=hidden)

    def forward(self, x, seq_mask_x, seq_len_x, y, z):

        # Encode the source sentence and initialize the decoder hidden state.
        encoder_outputs, encoder_final = self.encode(x, seq_len_x, z)
        hidden = self.init_decoder(encoder_outputs, encoder_final, z)

        # Estimate the Categorical parameters for E[P(x|z)] using the given sample of the latent
        # variable.
        lm_logits = self.run_language_model(x, z)

        if self.bow_output_layer is not None:
            bow_logits=self.bow_output_layer(z)
        else:
            bow_logits=None

        if self.bow_output_layer_tl is not None:
            bow_logits_tl=self.bow_output_layer_tl(z)
        else:
            bow_logits_tl=None


        # Estimate the Categorical parameters for E[P(y|x, z)] using the given sample of the latent
        # variable.
        tm_logits = []
        all_att_weights = []
        max_time = y.size(1)
        for t in range(max_time):
            prev_y = y[:, t]
            y_embed = self.tgt_embed(prev_y)
            pre_output, hidden, att_weights = self.decoder.step(y_embed, hidden, seq_mask_x,
                                                                encoder_outputs)
            logits = self.generate(pre_output)
            tm_logits.append(logits)
            all_att_weights.append(att_weights)

        return torch.cat(tm_logits, dim=1), lm_logits, torch.cat(all_att_weights, dim=1),bow_logits,bow_logits_tl

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

        # Estimate the Categorical parameters for E[P(x|z)] using the given sample of the latent
        # variable.
        # [max_length, batch_size, vocab_size]
        lm_logits = self.run_language_model(x_in, z)

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
        lm_logits = lm_logits.permute(0, 2, 1)
        tm_logits = tm_logits.permute(0, 2, 1)

        # [batch_size]
        tm_loss = F.cross_entropy(tm_logits, y_out, ignore_index=self.pad_idx, reduction="none").sum(dim=1)
        lm_loss = F.cross_entropy(lm_logits, x_out, ignore_index=self.pad_idx, reduction="none").sum(dim=1)

        return -lm_loss, -tm_loss

    def compute_lm_likelihood(self, x_in, seq_mask_x, seq_len_x, x_out, z):
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

        # Estimate the Categorical parameters for E[P(x|z)] using the given sample of the latent
        # variable.
        # [max_length, batch_size, vocab_size]
        lm_logits = self.run_language_model(x_in, z)
        # [batch_size, max_length, vocab_size]
        lm_logits = lm_logits.permute(0, 2, 1)

        lm_loss = F.cross_entropy(lm_logits, x_out, ignore_index=self.pad_idx, reduction="none").sum(dim=1)

        return -lm_loss

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
        tm_loss = F.cross_entropy(tm_logits, y_out, ignore_index=self.pad_idx, reduction="none").sum(dim=1)

        return -tm_loss

    def loss(self, tm_logits, lm_logits, targets_y, targets_x, qz, free_nats=0.,
             KL_weight=1., reduction="mean", bow_logits=None, bow_logits_tl=None):
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
        :param bow_logits: [B,vocab_size]
        """

        # Compute the loss for each batch element. Logits are of the form [B, T, vocab_size],
        # whereas the cross-entropy function wants a loss of the form [B, vocab_svocab_sizee, T].
        tm_logits = tm_logits.permute(0, 2, 1)
        tm_loss = F.cross_entropy(tm_logits, targets_y, ignore_index=self.pad_idx, reduction="none")
        tm_loss = tm_loss.sum(dim=1)

        # Compute the language model categorical loss.
        lm_loss = self.language_model.loss(lm_logits, targets_x, reduction="none")

        bow_loss=torch.zeros_like(lm_loss)
        if bow_logits is not None:
            bow_logprobs=-F.log_softmax(bow_logits,-1)
            bsz=bow_logits.size(0)
            for i in range(bsz):
                bow=torch.unique(targets_x[i])
                bow_mask=( bow != self.language_model.pad_idx)
                bow=bow.masked_select(bow_mask)
                bow_loss[i]=torch.sum( bow_logprobs[i][bow] )

        bow_loss_tl=torch.zeros_like(lm_loss)
        if bow_logits_tl is not None:
            bow_logprobs=-F.log_softmax(bow_logits_tl,-1)
            bsz=bow_logits_tl.size(0)
            for i in range(bsz):
                bow=torch.unique(targets_y[i])
                bow_mask=( bow != self.pad_idx)
                bow=bow.masked_select(bow_mask)
                bow_loss_tl[i]=torch.sum( bow_logprobs[i][bow] )

        # Compute the KL divergence between the distribution used to sample z, and the prior
        # distribution.
        pz = self.prior()  #.expand(qz.mean.size())

        # The loss is the negative ELBO.
        tm_log_likelihood = -tm_loss
        lm_log_likelihood = -lm_loss

        bow_log_likelihood = - bow_loss - bow_loss_tl


        # TODO: N this is [...,D], whereas with MoG this is [...]
        #  we need to wrap stuff around torch.distributions.Independent
        KL = torch.distributions.kl.kl_divergence(qz, pz)
        raw_KL = KL * 1
        #raw_KL = KL.sum(dim=1)
        #KL = KL.sum(dim=1)

        if free_nats > 0:
            KL = torch.clamp(KL, min=free_nats)
        KL *= KL_weight
        elbo = tm_log_likelihood + lm_log_likelihood + bow_log_likelihood - KL
        loss = -elbo

        out_dict = {
            'tm_log_likelihood': tm_log_likelihood,
            'lm_log_likelihood': lm_log_likelihood,
            'bow_log_likelihood': bow_log_likelihood,
            'KL': KL,
            'raw_KL': raw_KL
        }

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
