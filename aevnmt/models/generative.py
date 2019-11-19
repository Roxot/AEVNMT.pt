"""
AEVNMT has two generative components, namely, 
    a language model (LM) and a translation model (TM), 
    p(x|z) and p(y|z,x), respectively.

These are also known as decoders or generators, here we provide implementations for those.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Categorical, Bernoulli
from dgm.conditional import MADEConditioner
from dgm.likelihood import AutoregressiveLikelihood
from .rnnlm import RNNLM

from aevnmt.components.nibm1 import NeuralIBM1
from aevnmt.components import tile_rnn_hidden, rnn_creation_fn, tile_rnn_hidden_for_decoder


class GenerativeLM(nn.Module):
    """
    The forward will return the likelihoods 
        X_i|z,x_{<i}
    that is, a batch of sequences of torch distribution objects.

    To get the likelihood of an observation, use log_prob.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, z) -> Distribution:
        """
        Return distributions X_i|z, x_{<i}.

        x: [B, Tx] token ids
        z: [B, Dz] stochastic embeddings
        """
        raise NotImplementedError("Implement me!")

    def log_prob(self, likelihood: Distribution, x):
        """
        Return log-probability of observation.

        likelihood: as returned by forward
        x: [B, Tx] observed token ids
        """
        raise NotImplementedError("Implement me!")


class IndependentLM(GenerativeLM):
    """
    This draws source words from a single Categorical distribution
        P(x|z) = \prod_{i=1}^m Cat(x_i|f(z))
    where m = |x|.
    """

    def __init__(self, latent_size, vocab_size, pad_idx):
        super().__init__()
        self.pad_idx = pad_idx
        self.output_layer = nn.Linear(latent_size, vocab_size, bias=True)
    
    def forward(self, x, z) -> Categorical:
        """
        Return Categorical distributions
            X_i|z ~ Cat(f(z))
        with shape [B, 1, Vx]
        """
        # [B, 1, Vx]
        return Categorical(logits=self.output_layer(z).unsqueeze(1))
    
    def log_prob(self, likelihood: Categorical, x):
        # [B, Tx, Vx]
        likelihood = likelihood.expand(x.size())
        # [B, Tx]
        log_p = likelihood.log_prob(x)
        # mask invalid positions
        log_p = log_p * (x != self.pad_idx).float()
        # [B]
        log_p = log_p.sum(-1)
        return log_p

class CorrelatedBernoullisLM(GenerativeLM):
    """
    This parameterises an autoregressive product of Bernoulli distributions,
        P(x|z) = \prod_{v=1}^V Bern([v in x]|b_v(z, x))
    where V is the vocabulary size and b(z,x) \in (0, 1)^V is autoregressive in x (we use a MADE).
    """

    def __init__(self, vocab_size, latent_size, hidden_sizes, pad_idx, num_masks=10, resample_mask_every=10):
        super().__init__()
        self.pad_idx = pad_idx
        self.resample_every = resample_mask_every
        self.counter = resample_mask_every
        self.made_conditioner = MADEConditioner(
            input_size=vocab_size + latent_size, 
            output_size=vocab_size,  
            context_size=latent_size,
            hidden_sizes=hidden_sizes,
            num_masks=num_masks
        )
        self.product_of_bernoullis = AutoregressiveLikelihood(
            event_size=vocab_size, 
            dist_type=Bernoulli,
            conditioner=self.made_conditioner
        )

    def forward(self, x, z) -> Bernoulli:
        """
        Return Bernoulli distributions 
            [v \in X]|z, \Sigma_{<v} ~ Bernoulli(b_v(z, \Sigma_{<v}))
        with shape [B, Vx] where Vx = |\Sigma| and \Sigma is the vocabulary.
        """
        bsz = x.size(0)
        seq_mask_x = x != self.pad_idx
        made_input = torch.zeros((bsz, self.product_of_bernoullis.event_size), device=x.device)
        for i in range(bsz):
            bow = torch.unique(x[i] * seq_mask_x[i].type_as(x[i]))
            made_input[i][bow] = 1.0
        if self.resample_every > 0:
            self.counter = self.counter - 1 if self.counter > 0 else self.resample_every
        return self.product_of_bernoullis(z, history=made_input, resample_mask=self.resample_every > 0 and self.counter == 0)

    def log_prob(self, likelihood: Bernoulli, x):
        bsz = x.size(0)
        made_ref = torch.zeros((bsz,self.product_of_bernoullis.event_size),device=x.device)
        for i in range(bsz):
            bow = torch.unique(x[i])
            made_ref[i][bow] = 1.0
        return likelihood.log_prob(made_ref).sum(-1)

class CorrelatedCategoricalsLM(GenerativeLM):
    """
    This implements an autoregressive product of Categoricals likelihood, i.e.
        P(x|z) = \prod_{i=1}^m Cat(x_i|f(z, x_{<i})
    where m = |x|.
    """

    def __init__(self, embedder, latent_size, hidden_size, 
            dropout, num_layers, cell_type, tied_embeddings, feed_z, gate_z):  #TODO implement gate_z
        super().__init__()
        self.embedder = embedder
        self.pad_idx = embedder.padding_idx
        self.feed_z = feed_z
        self.hidden_size = hidden_size
        self.init_layer = nn.Sequential(
                nn.Linear(latent_size, hidden_size),
                nn.Tanh())
        rnn_dropout = 0. if num_layers == 1 else dropout
        rnn_fn = rnn_creation_fn(cell_type)
        feed_z_size = latent_size if feed_z else 0
        self.rnn = rnn_fn(embedder.embedding_dim + feed_z_size, hidden_size, 
            batch_first=True, dropout=rnn_dropout, num_layers=num_layers)
        self.tied_embeddings = tied_embeddings
        if not tied_embeddings:
            self.output_matrix = nn.Parameter(torch.randn(embedder.num_embeddings, hidden_size))
        self.dropout_layer = nn.Dropout(p=dropout)
    
    def step(self, x_embed, hidden, z):
        rnn_input = x_embed.unsqueeze(1)
        if self.feed_z:
            rnn_input=torch.cat([rnn_input, z.unsqueeze(1)], dim=-1)
        rnn_output, hidden = self.rnn(rnn_input, hidden)
        rnn_output = self.dropout_layer(rnn_output)
        W_out = self.embedder.weight if self.tied_embeddings else self.output_matrix
        logits = F.linear(rnn_output, W_out)
        return hidden, logits

    def unroll(self, x, hidden, z):
        # [B, Tx, Dx]
        x_embed = self.dropout_layer(self.embedder(x))
        outputs = []
        for t in range(x_embed.size(1)):
            # [B, 1, Dx]
            rnn_input = x_embed[:, t].unsqueeze(1)
            if self.feed_z:
                rnn_input=torch.cat([rnn_input, z.unsqueeze(1)],dim=-1)
            rnn_output, hidden = self.rnn(rnn_input, hidden)
            rnn_output = self.dropout_layer(rnn_output)
            W_out = self.embedder.weight if self.tied_embeddings else self.output_matrix
            logits = F.linear(rnn_output, W_out)
            outputs.append(logits)
        return torch.cat(outputs, dim=1)

    def forward(self, x, z) -> Categorical:
        """
        Return Categorical distributions
            X_i|z, x_{<i} ~ Cat(f(z, x_{<i}))
        with shape [B, Tx, Vx]
        """
        hidden = tile_rnn_hidden(self.init_layer(z), self.rnn)
        # [B, Tx, Vx]
        logits = self.unroll(x, hidden=hidden, z=z) 
        return Categorical(logits=logits)

    def log_prob(self, likelihood: Categorical, x):
        # [B, Tx] -> [B]
        return (likelihood.log_prob(x) * (x != self.pad_idx)).sum(-1)

class GenerativeTM(nn.Module):
    """
    The forward will return the likelihoods 
        Y_j|z,x, y_{<j}
    that is, a batch of sequences of torch distribution objects.

    To get the likelihood of an observation, use log_prob.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, seq_mask_x, seq_len_x, y, z) -> Distribution:
        """
        Return distributions Y_j|z,x,y_{<j}.

        x: [B, Tx] source token ids
        seq_mask_x: [B, Tx] 1 if token is not PAD, 0 otherwise
        seq_len_x: [B] length of source sequences
        y: [B, Ty] target token ids
        z: [B, Dz] stochastic embedding
        """
        raise NotImplementedError("Implement me!")

    def log_prob(self, likelihood: Distribution, y):
        """
        Return log-probability of observation.

        likelihood: as returned by forward
        y: [B, Ty] observed token ids
        """
        raise NotImplementedError("Implement me!")


class IndependentTM(GenerativeTM):
    """
    This draws target words from a single Categorical distribution
        P(y|z,x) = \prod_{j=1}^n Cat(y_j|f(z))
    where n = |y|. Note that the parameterisation currently ignores x.
    """

    def __init__(self, latent_size, vocab_size, pad_idx):
        super().__init__()
        self.tgt_independent_lm = IndependentLM(
            latent_size=latent_size, 
            vocab_size=vocab_size, 
            pad_idx=pad_idx)
    
    def forward(self, x, seq_mask_x, seq_len_x, y, z) -> Categorical:
        return self.tgt_independent_lm(y, z)
    
    def log_prob(self, likelihood: Categorical, y):
        return self.tgt_independent_lm.log_prob(likelihood, y)


class CorrelatedBernoullisTM(GenerativeTM):
    """

    P(x|z) = \prod_{v=1}^V Bern([v in x]|b_v(z, x))

    where V is the vocabulary size and b(z,x) \in (0, 1)^V is autoregressive in x (we use a MADE).
    """

    def __init__(self, vocab_size, latent_size, hidden_sizes, pad_idx, num_masks=10, resample_mask_every=10):
        super().__init__()
        self.correlated_bernoullis_lm = CorrelatedBernoullisLM(
            vocab_size=vocab_size,
            latent_size=latent_size,
            hidden_sizes=hidden_sizes,
            pad_idx=pad_idx,
            num_masks=num_masks,
            resample_mask_every=resample_mask_every
        )

    def forward(self, x, seq_mask_x, seq_len_x, y, z) -> Bernoulli:
        return self.correlated_bernoullis_lm(y, z)

    def log_prob(self, likelihood: Bernoulli, y):
        return self.correlated_bernoullis_lm.log_prob(likelihood, y)

class CorrelatedCategoricalsTM(GenerativeTM):
    """
    This implements an autoregressive product of Categoricals
        P(y|z,x) = \prod_{j=1}^n Cat(y_j|f(z, y_{<j}))
    where n = |y|. Note that for now the parameterisation ignores x.
    """

    def __init__(self, embedder, latent_size, hidden_size, 
            dropout, num_layers, cell_type, tied_embeddings, feed_z, gate_z):
        super().__init__()
        self.correlated_categoricals_lm = CorrelatedCategoricalsLM(
            embedder=embedder,
            latent_size=latent_size,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=num_layers,
            cell_type=cell_type,
            tied_embeddings=tied_embeddings,
            feed_z=feed_z,
            gate_z=gate_z
        )

    def forward(self, x, seq_mask_x, seq_len_x, y, z) -> Categorical:
        return self.correlated_categoricals_lm(y, z)

    def log_prob(self, likelihood: Categorical, y):
        return self.correlated_categoricals_lm.log_prob(likelihood, y)

class IBM1TM(GenerativeTM):
    """
    This implements an IBM1-style likelihood
        P(y|x,z) = \prod_{j=1}^n \sum_{i=1}^m P(a_j=i)P(y_j|x_i, z)
    where n = |y|, m = |x|, P(a_j) = 1/m, P(y_j|x_i, z) = Cat(y_j|f(z * g(x_i)))
    and g(x_i) gates the stochastic embedding z.

    We intentionally downplay the role of x to promote use of z.
    """

    def __init__(self, src_embed, latent_size, hidden_size, src_vocab_size, tgt_vocab_size, pad_idx):
        super().__init__()
        self.pad_idx = pad_idx
        self.src_embed = src_embed
        self.gate = nn.Sequential(
            nn.Linear(src_embed.embedding_dim + latent_size, latent_size),
            nn.Tanh(),
            nn.Linear(latent_size, latent_size),
            nn.Sigmoid()
        )
        self.proj = nn.Sequential(nn.Linear(latent_size, latent_size), nn.Tanh())
        self.nibm1 = NeuralIBM1(src_vocab_size, tgt_vocab_size, latent_size, hidden_size, pad_idx)

    def forward(self, x, seq_mask_x, seq_len_x, y, z) -> Categorical:
        """
        Return Categorical distributions
            Y_j|z,x
        (see documentation of class)
        with shape [B, Ty, Vy]
        """
        # [B, Tx, Dx]
        x = self.src_embed(x)
        # [B, Tx, Dz]
        g = self.gate(torch.cat([x, z.unsqueeze(1).repeat([1, x.size(1), 1])], -1))
        # [B, Tx, Dz]
        x = self.proj(g * z.unsqueeze(1))
        # [B, Ty, V]
        marginals = - self.nibm1(x, seq_mask_x, seq_len_x, y.size(1))
        return Categorical(probs=marginals)

    def log_prob(self, likelihood: Categorical, y):
        return (likelihood.log_prob(y) * (y != self.pad_idx).float()).sum(-1)


class AttentionBasedTM(GenerativeTM):

    def __init__(self, src_embedder, tgt_embedder, encoder, decoder, latent_size, dropout, feed_z, tied_embeddings):
        super().__init__()
        self.src_embedder = src_embedder
        self.tgt_embedder = tgt_embedder
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_init_layer = nn.Sequential(
            nn.Linear(latent_size, encoder.hidden_size),
            nn.Tanh())
        self.decoder_init_layer = nn.Sequential(
            nn.Linear(latent_size, decoder.hidden_size),
            nn.Tanh())
        self.tied_embeddings = tied_embeddings
        if not tied_embeddings:
            self.output_matrix = nn.Parameter(torch.randn(tgt_embedder.num_embeddings, decoder.hidden_size))
        self.dropout_layer = nn.Dropout(p=dropout)
        self.feed_z = feed_z
    
    def src_embed(self, x):
        x_embed = self.src_embedder(x)
        x_embed = self.dropout_layer(x_embed)
        return x_embed
    
    def tgt_embed(self, y):
        y_embed = self.tgt_embedder(y)
        y_embed = self.dropout_layer(y_embed)
        return y_embed

    def init_decoder(self, encoder_outputs, encoder_final, z):
        self.decoder.init_decoder(encoder_outputs, encoder_final)
        hidden = tile_rnn_hidden_for_decoder(self.decoder_init_layer(z), self.decoder)
        return hidden

    def encode(self, x, seq_len_x, z):
        x_embed = self.src_embed(x)
        hidden = tile_rnn_hidden(self.encoder_init_layer(z), self.encoder.rnn)
        return self.encoder(x_embed, seq_len_x, hidden=hidden)

    def generate(self, pre_output):
        W = self.tgt_embedder.weight if self.tied_embeddings else self.output_matrix
        return F.linear(pre_output, W)

    def forward(self, x, seq_mask_x, seq_len_x, y, z, state=dict()) -> Categorical:
        encoder_outputs, encoder_final = self.encode(x, seq_len_x, z)
        hidden = self.init_decoder(encoder_outputs, encoder_final, z)

        # Estimate the Categorical parameters for E[P(y|x, z)] using the given sample of the latent
        # variable.
        tm_logits = []
        all_att_weights = []
        max_time = y.size(1)
        for t in range(max_time):
            prev_y = y[:, t]
            y_embed = self.tgt_embed(prev_y)
            pre_output, hidden, att_weights = self.decoder.step(
                y_embed, hidden, seq_mask_x, encoder_outputs, z=z if self.feed_z else None)
            logits = self.generate(pre_output)
            tm_logits.append(logits)
            all_att_weights.append(att_weights)
        state['all_att_weights'] = torch.cat(all_att_weights, dim=1)
        # [B, Ty, Vy]
        return Categorical(torch.cat(tm_logits, dim=1))

    def log_prob(self, likelihood: Categorical, y):
        return (likelihood.log_prob(y) * (y != self.tgt_embedder.padding_idx).float()).sum(-1)
