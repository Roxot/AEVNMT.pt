import torch
import torch.nn as nn
import torch.nn.functional as F

from aevnmt.components import RNNEncoder, tile_rnn_hidden, tile_rnn_hidden_for_decoder
from aevnmt.dist import NormalLayer

from itertools import chain

class InferenceNetwork(nn.Module):

    def __init__(self, src_embedder, hidden_size, latent_size, bidirectional, num_enc_layers, cell_type):
        """
        :param src_embedder: uses this embedder, but detaches its output from the graph as to not compute
                             gradients for it.
        """
        super().__init__()
        self.src_embedder = src_embedder
        emb_size = src_embedder.embedding_dim
        self.encoder = RNNEncoder(emb_size=emb_size,
                                  hidden_size=hidden_size,
                                  bidirectional=bidirectional,
                                  dropout=0.,
                                  num_layers=num_enc_layers,
                                  cell_type=cell_type)
        encoding_size = hidden_size if not bidirectional else hidden_size * 2
        self.normal_layer = NormalLayer(encoding_size, hidden_size, latent_size)

    def forward(self, x, seq_mask_x, seq_len_x):
        x_embed = self.src_embedder(x).detach()
        encoder_outputs, _ = self.encoder(x_embed, seq_len_x)
        avg_encoder_output = (encoder_outputs * seq_mask_x.unsqueeze(-1).type_as(encoder_outputs)).sum(dim=1)
        return self.normal_layer(avg_encoder_output)

    def parameters(self, recurse=True):
        return chain(self.encoder.parameters(recurse=recurse), self.normal_layer.parameters(recurse=recurse))

    def named_parameters(self, prefix='', recurse=True):
        return chain(self.encoder.named_parameters(prefix='', recurse=True), self.normal_layer.named_parameters(prefix='', recurse=True), )

class AEVNMT(nn.Module):

    def __init__(self, tgt_vocab_size, emb_size, latent_size, encoder, decoder, language_model,
                 pad_idx, dropout, tied_embeddings):
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
        self.inf_network = InferenceNetwork(src_embedder=self.language_model.embedder,
                                            hidden_size=encoder.hidden_size,
                                            latent_size=latent_size,
                                            bidirectional=encoder.bidirectional,
                                            num_enc_layers=encoder.num_layers,
                                            cell_type=encoder.cell_type)

        # This is done because the location and scale of the prior distribution are not considered
        # parameters, but are rather constant. Registering them as buffers still makes sure that
        # they will be moved to the appropriate device on which the model is run.
        self.register_buffer("prior_loc", torch.zeros([latent_size]))
        self.register_buffer("prior_scale", torch.ones([latent_size]))

    def inference_parameters(self):
        return self.inf_network.parameters()

    def generative_parameters(self):
        # TODO: separate the generative model into a GenerativeModel module
        #  within that module, have two modules, namely, LanguageModel and TranslationModel
        return chain(self.lm_parameters(), self.tm_parameters())

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

    def approximate_posterior(self, x, seq_mask_x, seq_len_x):
        """
        Returns an approximate posterior distribution q(z|x).
        """
        return self.inf_network(x, seq_mask_x, seq_len_x)

    def prior(self):
        return torch.distributions.Normal(loc=self.prior_loc,
                                          scale=self.prior_scale)

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

        return torch.cat(tm_logits, dim=1), lm_logits, torch.cat(all_att_weights, dim=1)

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
             KL_weight=1., reduction="mean"):
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
        """

        # Compute the loss for each batch element. Logits are of the form [B, T, vocab_size],
        # whereas the cross-entropy function wants a loss of the form [B, vocab_svocab_sizee, T].
        tm_logits = tm_logits.permute(0, 2, 1)
        tm_loss = F.cross_entropy(tm_logits, targets_y, ignore_index=self.pad_idx, reduction="none")
        tm_loss = tm_loss.sum(dim=1)

        # Compute the language model categorical loss.
        lm_loss = self.language_model.loss(lm_logits, targets_x, reduction="none")

        # Compute the KL divergence between the distribution used to sample z, and the prior
        # distribution.
        pz = self.prior().expand(qz.mean.size())

        # The loss is the negative ELBO.
        tm_log_likelihood = -tm_loss
        lm_log_likelihood = -lm_loss

        KL = torch.distributions.kl.kl_divergence(qz, pz)
        raw_KL = KL.sum(dim=1)
        KL = KL.sum(dim=1)

        if free_nats > 0:
            KL = torch.clamp(KL, min=free_nats)
        KL *= KL_weight
        elbo = tm_log_likelihood + lm_log_likelihood - KL
        loss = -elbo

        out_dict = {
            'tm_log_likelihood': tm_log_likelihood,
            'lm_log_likelihood': lm_log_likelihood,
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
