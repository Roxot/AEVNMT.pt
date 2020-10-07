"""
AEVNMT has two generative components, namely, 
    a language model (LM) and a translation model (TM), 
    p(x|z) and p(y|z,x), respectively.

These are also known as decoders or generators, here we provide implementations for those.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Categorical, Bernoulli, Poisson
from probabll.dgm import register_conditional_parameterization
from probabll.dgm.conditional import MADEConditioner
from probabll.dgm.likelihood import AutoregressiveLikelihood
from .rnnlm import RNNLM

from aevnmt.components.nibm1 import NeuralIBM1
from aevnmt.components import tile_rnn_hidden, rnn_creation_fn, tile_rnn_hidden_for_decoder
from aevnmt.components import TransformerEncoder, TransformerDecoder
from aevnmt.components.transformer import generate_padding_mask, generate_square_subsequent_mask


class GenerativeLM(nn.Module):
    """
    The forward will return the likelihoods 
        X_i|z,x_{<i}
    that is, a batch of sequences of torch distribution objects.

    To get the likelihood of an observation, use log_prob.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, z, state=dict()) -> Distribution:
        """
        Return distributions X_i|z, x_{<i}.

        x: [B, Tx] token ids
        z: [B, Dz] stochastic embeddings
        state: use this dictionary to return something about the forward pass (for example, 
            attention weights, or other details of the computation that you whish to log)
        """
        raise NotImplementedError("Implement me!")

    def log_prob(self, likelihood: Distribution, x):
        """
        Return log-probability of observation.

        likelihood: as returned by forward
        x: [B, Tx] observed token ids
        """
        raise NotImplementedError("Implement me!")

    def sample(self, z, max_len=100, greedy=False, state=dict()):
        """
        Sample from X|z where z [B, Dz]
        """
        raise NotImplementedError("Implement me!")


class IndependentLM(GenerativeLM):
    """
    This draws source words from a single Categorical distribution
        P(x|z) = \prod_{i=1}^m Cat(x_i|f(z))
    where m = |x|.
    """

    def __init__(self, latent_size, embedder, tied_embeddings=True, dropout=0.5):
        super().__init__()
        vocab_size = embedder.num_embeddings
        self.pad_idx = embedder.padding_idx

        if tied_embeddings:
            self.encoder = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(latent_size, embedder.embedding_dim, bias=False),
                    nn.Tanh(),
                    nn.Dropout(dropout)
            )
            self.output_matrix = embedder.weight  # [V, Dx]
        else:
            self.encoder = nn.Dropout(dropout)
            self.output_matrix = nn.Parameter(torch.randn(vocab_size, latent_size))  # [V, Dz]
    
    def forward(self, x, z, state=dict()) -> Categorical:
        """
        Return Categorical distributions
            X_i|z ~ Cat(f(z))
        with shape [B, 1, Vx]
        """
        # [B, Vx]
        logits = F.linear(self.encoder(z), self.output_matrix)
        # [B, 1, Vx]
        return Categorical(logits=logits.unsqueeze(1))
    
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
    
    def sample(self, z, max_len=100, greedy=False, state=dict()):
        """
        Sample from X|z where z [B, Dz]
        """
        raise NotImplementedError("Implement me!")
        likelihood = self(None, z)  # TODO deal with max_len
        if greedy:
            x = torch.argmax(likelihood.logits, dim=-1)
        else:
            x = likelihood.sample()
        return x

class CorrelatedBernoullisLM(GenerativeLM):
    """
    This parameterises an autoregressive product of Bernoulli distributions,
        P(x|z) = \prod_{v=1}^V Bern([v in x]|b_v(z, x))
    where V is the vocabulary size and b(z,x) \in (0, 1)^V is autoregressive in x (we use a MADE).
    """

    def __init__(self, vocab_size, latent_size, hidden_sizes, pad_idx, num_masks=1, resample_mask_every=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.resample_every = resample_mask_every
        self.counter = resample_mask_every
        self.vocab_size = vocab_size
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

    def make_indicators(self, x):
        """Return a vocab_size-dimensional bit-vector view of x"""
        # We convert ids to V-dimensional one-hot vectors and reduce-sum the time dimension
        #  this gives us word counts 
        # [B, T] -> [B, T, V] -> [B, V]
        word_counts = F.one_hot(x, self.vocab_size).sum(1) 
        word_counts[:, self.pad_idx] = 0
        indicators = (word_counts > 0).float()
        return indicators

    def forward(self, x, z, state=dict()) -> Bernoulli:
        """
        Return Bernoulli distributions 
            [v \in X]|z, \Sigma_{<v} ~ Bernoulli(b_v(z, \Sigma_{<v}))
        with shape [B, Vx] where Vx = |\Sigma| and \Sigma is the vocabulary.
        """
        # We convert ids to V-dimensional one-hot vectors and sum the time dimension
        #  this gives us word counts 
        # [B, V]
        indicators = self.make_indicators(x)
        if self.resample_every > 0:
            self.counter = self.counter - 1 if self.counter > 0 else self.resample_every
        return self.product_of_bernoullis(z, history=indicators, resample_mask=self.resample_every > 0 and self.counter == 0)

    def log_prob(self, likelihood: Bernoulli, x):
        # [B, V]
        indicators = self.make_indicators(x)
        # [B, V] -> [B]
        return likelihood.log_prob(indicators).sum(-1)
    
    def sample(self, z, max_len=None, greedy=False, state=dict()):
        """
        Sample from X|z where z [B, Dz]
        """
        shape = [z.size(0), self.product_of_bernoullis.event_size]
        if greedy:
            raise NotImplementedError("Greedy decoding not implemented for MADE")
        x = self.product_of_bernoullis.sample(z, torch.zeros(shape, dtype=z.dtype, device=z.device))
        return x


@register_conditional_parameterization(Poisson)
def make_poisson(inputs, event_size):
    assert inputs.size(-1) == event_size, "Expected [...,%d] got [...,%d]" % (event_size, inputs.size(-1))
    # we clamp the Poisson rate to [1e-6, 30] to prevent instabilities 
    # this is relatively mild (as we don't expect many words to be repeat 30 times or more on average)
    return Poisson(torch.clamp(F.softplus(inputs), min=1e-6, max=30))


class CorrelatedPoissonsLM(GenerativeLM):
    """
    This parameterises an autoregressive product of Poisson distributions,
        P(x|z) = \prod_{v=1}^V Bern(c_v(x)|b_v(z, x))
    where V is the vocabulary size, c_v(x) counts the occurrences of v in x,
    and b(z,x) \in (0, infty)^V is autoregressive in x (we use a MADE).
    """

    def __init__(self, vocab_size, latent_size, hidden_sizes, pad_idx, num_masks=1, resample_mask_every=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.resample_every = resample_mask_every
        self.counter = resample_mask_every
        self.vocab_size = vocab_size
        self.made_conditioner = MADEConditioner(
            input_size=vocab_size + latent_size, 
            output_size=vocab_size,  
            context_size=latent_size,
            hidden_sizes=hidden_sizes,
            num_masks=num_masks
        )
        self.product_of_poissons = AutoregressiveLikelihood(
            event_size=vocab_size, 
            dist_type=Poisson,
            conditioner=self.made_conditioner
        )

    def make_counts(self, x):
        """Return a vocab_size-dimensional count-vector view of x"""
        # We convert ids to V-dimensional one-hot vectors and reduce-sum the time dimension
        #  this gives us word counts 
        # [B, T] -> [B, T, V] -> [B, V]
        word_counts = F.one_hot(x, self.vocab_size).sum(1) 
        word_counts[:, self.pad_idx] = 0  # we could actually leave it here, it is a way to model length
        return word_counts.float()

    def forward(self, x, z, state=dict()) -> Poisson:
        """
        Return Poisson distributions 
            c_v(X)|z, \Sigma_{<v} ~ Poisson(b_v(z, \Sigma_{<v}))
        with shape [B, Vx] where Vx = |\Sigma| and \Sigma is the vocabulary.
        """
        # We convert ids to V-dimensional one-hot vectors and sum the time dimension
        #  this gives us word counts 
        # [B, V]
        counts = self.make_counts(x)
        if self.resample_every > 0:
            self.counter = self.counter - 1 if self.counter > 0 else self.resample_every
        return self.product_of_poissons(z, history=counts, resample_mask=self.resample_every > 0 and self.counter == 0)

    def log_prob(self, likelihood: Poisson, x):
        # [B, V]
        counts = self.make_counts(x)
        # [B, V] -> [B]
        return likelihood.log_prob(counts).sum(-1)
    
    def sample(self, z, max_len=None, greedy=False, state=dict()):
        """
        Sample from X|z where z [B, Dz]
        """
        shape = [z.size(0), self.product_of_poissons.event_size]
        if greedy:
            raise NotImplementedError("Greedy decoding not implemented for MADE")
        x = self.product_of_poissons.sample(z, torch.zeros(shape, dtype=z.dtype, device=z.device))
        return x


class CorrelatedCategoricalsLM(GenerativeLM):
    """
    This implements an autoregressive product of Categoricals likelihood, i.e.
        P(x|z) = \prod_{i=1}^m Cat(x_i|f(z, x_{<i})
    where m = |x|.
    """

    def __init__(self, embedder, sos_idx, eos_idx, latent_size, hidden_size, 
            dropout, num_layers, cell_type, tied_embeddings, feed_z, gate_z):  #TODO implement gate_z
        super().__init__()
        self.embedder = embedder
        self.pad_idx = embedder.padding_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
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
    
    def init(self, z):
        hidden = tile_rnn_hidden(self.init_layer(z), self.rnn)
        return hidden

    def generate(self, pre_output):
        W = self.embedder.weight if self.tied_embeddings else self.output_matrix
        return F.linear(pre_output, W)

    def step(self, x_embed, hidden, z):
        rnn_input = x_embed.unsqueeze(1)
        if self.feed_z:
            rnn_input=torch.cat([rnn_input, z.unsqueeze(1)], dim=-1)
        rnn_output, hidden = self.rnn(rnn_input, hidden)
        rnn_output = self.dropout_layer(rnn_output)
        return hidden, rnn_output

    def unroll(self, x, hidden, z):
        # [B, Tx, Dx]
        x_embed = self.dropout_layer(self.embedder(x))
        outputs = []
        for t in range(x_embed.size(1)):
            hidden, rnn_output = self.step(x_embed[:, t], hidden, z)
            logits = self.generate(rnn_output)
            outputs.append(logits)
        return torch.cat(outputs, dim=1)

    def forward(self, x, z, state=dict()) -> Categorical:
        """
        Return Categorical distributions
            X_i|z, x_{<i} ~ Cat(f(z, x_{<i}))
        with shape [B, Tx, Vx]
        """
        hidden = self.init(z)
        # [B, Tx, Vx]
        logits = self.unroll(x, hidden=hidden, z=z) 
        return Categorical(logits=logits)

    def log_prob(self, likelihood: Categorical, x):
        # [B, Tx] -> [B]
        return (likelihood.log_prob(x) * (x != self.pad_idx).float()).sum(-1)
    
    def sample(self, z, max_len=100, greedy=False, state=dict()):
        """
        Sample from X|z where z [B, Dz]
        """
        batch_size = z.size(0)
        hidden = self.init(z)
        prev_y = torch.full(size=[batch_size], fill_value=self.sos_idx, dtype=torch.long,
            device=self.embedder.weight.device)

        # Decode step-by-step by picking the maximum probability word
        # at each time step.
        predictions = []
        log_probs = []
        is_complete = torch.zeros_like(prev_y).unsqueeze(-1).byte()
        for t in range(max_len):
            prev_y = self.embedder(prev_y)
            hidden, pre_output = self.step(prev_y, hidden, z)
            logits = self.generate(pre_output)
            px_z = Categorical(logits=logits)
            if greedy:
                prediction = torch.argmax(logits, dim=-1)
            else:
                prediction = px_z.sample()
            prev_y = prediction.view(batch_size)
            log_prob_pred = px_z.log_prob(prediction)
            log_probs.append(torch.where(is_complete, torch.zeros_like(log_prob_pred), log_prob_pred))
            predictions.append(torch.where(is_complete, torch.full_like(prediction, self.embedder.padding_idx), prediction))
            is_complete = is_complete | (prediction == self.eos_idx).byte()

        state['log_prob'] = torch.cat(log_probs, dim=1)
        return torch.cat(predictions, dim=1)

class GenerativeTM(nn.Module):
    """
    The forward will return the likelihoods 
        Y_j|z,x, y_{<j}
    that is, a batch of sequences of torch distribution objects.

    To get the likelihood of an observation, use log_prob.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, seq_mask_x, seq_len_x, y, z, state=dict()) -> Distribution:
        """
        Return distributions Y_j|z,x,y_{<j}.

        x: [B, Tx] source token ids
        seq_mask_x: [B, Tx] 1 if token is not PAD, 0 otherwise
        seq_len_x: [B] length of source sequences
        y: [B, Ty] target token ids
        z: [B, Dz] stochastic embedding
        state: use this dictionary to return something about the forward pass (for example, 
            attention weights, or other details of the computation that you whish to log)
        """
        raise NotImplementedError("Implement me!")

    def log_prob(self, likelihood: Distribution, y):
        """
        Return log-probability of observation.

        likelihood: as returned by forward
        y: [B, Ty] observed token ids
        """
        raise NotImplementedError("Implement me!")
    
    def sample(self, x, seq_mask_x, seq_len_x, z, max_len=None, greedy=False, state=dict()):
        raise NotImplementedError("Implement me!")


class IndependentTM(GenerativeTM):
    """
    This draws target words from a single Categorical distribution
        P(y|z,x) = \prod_{j=1}^n Cat(y_j|f(z))
    where n = |y|. Note that the parameterisation currently ignores x.
    """

    def __init__(self, latent_size, embedder, tied_embeddings):
        super().__init__()
        self.tgt_independent_lm = IndependentLM(
            latent_size=latent_size, embedder=embedder, tied_embeddings=tied_embeddings)
    
    def forward(self, x, seq_mask_x, seq_len_x, y, z, state=dict()) -> Categorical:
        return self.tgt_independent_lm(y, z)
    
    def log_prob(self, likelihood: Categorical, y):
        return self.tgt_independent_lm.log_prob(likelihood, y)
    
    def sample(self, x, seq_mask_x, seq_len_x, z, max_len=100, greedy=False, state=dict()):
        return self.tgt_independent_lm(z, max_len=max_len, greedy=greedy, state=state)


class CorrelatedBernoullisTM(GenerativeTM):
    """

    P(y|z,x) = \prod_{v=1}^V Bern([v in y]|b_v(z, y))

    where V is the vocabulary size and b(z,x) \in (0, 1)^V is autoregressive in y (we use a MADE).
    Note that for now the parameterisation ignores x.
    """

    def __init__(self, vocab_size, latent_size, hidden_sizes, pad_idx, num_masks=1, resample_mask_every=0):
        super().__init__()
        self.correlated_bernoullis_lm = CorrelatedBernoullisLM(
            vocab_size=vocab_size,
            latent_size=latent_size,
            hidden_sizes=hidden_sizes,
            pad_idx=pad_idx,
            num_masks=num_masks,
            resample_mask_every=resample_mask_every
        )

    def forward(self, x, seq_mask_x, seq_len_x, y, z, state=dict()) -> Bernoulli:
        return self.correlated_bernoullis_lm(y, z)

    def log_prob(self, likelihood: Bernoulli, y):
        return self.correlated_bernoullis_lm.log_prob(likelihood, y)
    
    def sample(self, x, seq_mask_x, seq_len_x, z, max_len=None, greedy=False, state=dict()):
        return self.correlated_bernoullis_lm(z, max_len=max_len, greedy=greedy, state=state)


class CorrelatedPoissonsTM(GenerativeTM):
    """

    P(y|z,x) = \prod_{v=1}^V Poisson(c_v(y)|b_v(z, y))

    where V is the vocabulary size, c_v(y) returns the number of occurrences of v in y,
        and b(z,x) \in (0, infty)^V is autoregressive in y (we use a MADE).

    Note that for now the parameterisation ignores x.
    """

    def __init__(self, vocab_size, latent_size, hidden_sizes, pad_idx, num_masks=1, resample_mask_every=0):
        super().__init__()
        self.correlated_poissons_lm = CorrelatedPoissonsLM(
            vocab_size=vocab_size,
            latent_size=latent_size,
            hidden_sizes=hidden_sizes,
            pad_idx=pad_idx,
            num_masks=num_masks,
            resample_mask_every=resample_mask_every
        )

    def forward(self, x, seq_mask_x, seq_len_x, y, z, state=dict()) -> Poisson:
        return self.correlated_poissons_lm(y, z)

    def log_prob(self, likelihood: Poisson, y):
        return self.correlated_poissons_lm.log_prob(likelihood, y)
    
    def sample(self, x, seq_mask_x, seq_len_x, z, max_len=None, greedy=False, state=dict()):
        return self.correlated_poissons_lm(z, max_len=max_len, greedy=greedy, state=state)

class CorrelatedCategoricalsTM(GenerativeTM):
    """
    This implements an autoregressive product of Categoricals
        P(y|z,x) = \prod_{j=1}^n Cat(y_j|f(z, y_{<j}))
    where n = |y|. Note that for now the parameterisation ignores x.
    """

    def __init__(self, embedder, sos_idx, eos_idx, latent_size, hidden_size, 
            dropout, num_layers, cell_type, tied_embeddings, feed_z, gate_z):
        super().__init__()
        self.correlated_categoricals_lm = CorrelatedCategoricalsLM(
            embedder=embedder,
            sos_idx=sos_idx,
            eos_idx=eos_idx,
            latent_size=latent_size,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=num_layers,
            cell_type=cell_type,
            tied_embeddings=tied_embeddings,
            feed_z=feed_z,
            gate_z=gate_z
        )

    def forward(self, x, seq_mask_x, seq_len_x, y, z, state=dict()) -> Categorical:
        return self.correlated_categoricals_lm(y, z)

    def log_prob(self, likelihood: Categorical, y):
        return self.correlated_categoricals_lm.log_prob(likelihood, y)
    
    def sample(self, x, seq_mask_x, seq_len_x, z, max_len=None, greedy=False, state=dict()):
        return self.correlated_poissons_lm(z, max_len=max_len, greedy=greedy, state=state)

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

    def forward(self, x, seq_mask_x, seq_len_x, y, z, state=dict()) -> Categorical:
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

    def __init__(self, src_embedder, tgt_embedder, tgt_sos_idx, tgt_eos_idx,
            encoder, decoder, latent_size, dropout, feed_z, tied_embeddings):
        # TODO Currently, if latent_size == 0, the model does not condition on latent_size (for NMT).
        # Change feed_z like in TransformerTM so it has this functionality (for example: feed_z_method=['enc_init', 'dec_init', 'cat'])
        super().__init__()
        self.src_embedder = src_embedder
        self.tgt_embedder = tgt_embedder
        self.tgt_sos_idx = tgt_sos_idx
        self.tgt_eos_idx = tgt_eos_idx
        self.encoder = encoder
        self.decoder = decoder
        
        self.tied_embeddings = tied_embeddings
        if not tied_embeddings:
            self.output_matrix = nn.Parameter(torch.randn(tgt_embedder.num_embeddings, decoder.hidden_size))
        self.dropout_layer = nn.Dropout(p=dropout)

        self.feed_z = feed_z
        self.latent_size = latent_size
        if latent_size > 0:
            self.encoder_init_layer = nn.Sequential(
                nn.Linear(latent_size, encoder.hidden_size),
                nn.Tanh())
            self.decoder_init_layer = nn.Sequential(
                nn.Linear(latent_size, decoder.hidden_size),
                nn.Tanh())
    
    def src_embed(self, x):
        x_embed = self.src_embedder(x)
        x_embed = self.dropout_layer(x_embed)
        return x_embed
    
    def tgt_embed(self, y):
        y_embed = self.tgt_embedder(y)
        y_embed = self.dropout_layer(y_embed)
        return y_embed

    def init_decoder(self, encoder_outputs, encoder_final, z):
        if self.latent_size:
            self.decoder.init_decoder(encoder_outputs, encoder_final)
            hidden = tile_rnn_hidden_for_decoder(self.decoder_init_layer(z), self.decoder)
        else:
            hidden = self.decoder.init_decoder(encoder_outputs, encoder_final)
        return hidden

    def encode(self, x, seq_len_x, z):
        x_embed = self.src_embed(x)
        if self.latent_size:
            hidden = tile_rnn_hidden(self.encoder_init_layer(z), self.encoder.rnn)
        else:
            hidden = None
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
        state['att_weights'] = torch.cat(all_att_weights, dim=1)
        # [B, Ty, Vy]
        return Categorical(logits=torch.cat(tm_logits, dim=1))

    def log_prob(self, likelihood: Categorical, y):
        return (likelihood.log_prob(y) * (y != self.tgt_embedder.padding_idx).float()).sum(-1)

    def sample(self, x, seq_mask_x, seq_len_x, z, max_len=100, greedy=False, state=dict()):
        encoder_outputs, encoder_final = self.encode(x, seq_len_x, z)
        hidden = self.init_decoder(encoder_outputs, encoder_final, z)
        batch_size = seq_mask_x.size(0)
        prev_y = torch.full(size=[batch_size], fill_value=self.tgt_sos_idx, dtype=torch.long, device=seq_mask_x.device)

        # Decode step-by-step by picking the maximum probability word
        # at each time step.
        predictions = []
        log_probs = []
        is_complete = torch.zeros_like(prev_y).unsqueeze(-1).byte()
        for t in range(max_len):
            prev_y = self.tgt_embedder(prev_y)
            pre_output, hidden, _ = self.decoder.step(
                prev_y, hidden, seq_mask_x, encoder_outputs, z=z if self.feed_z else None)
            logits = self.generate(pre_output)
            py_xz = Categorical(logits=logits)
            if greedy:
                prediction = torch.argmax(logits, dim=-1)
            else:
                prediction = py_xz.sample()
            prev_y = prediction.view(batch_size)
            log_prob_pred = py_xz.log_prob(prediction)
            log_probs.append(torch.where(is_complete, torch.zeros_like(log_prob_pred), log_prob_pred))
            predictions.append(torch.where(is_complete, torch.full_like(prediction, self.tgt_embedder.padding_idx), prediction))
            is_complete = is_complete | (prediction == self.tgt_eos_idx).byte()

        state['log_prob'] = torch.cat(log_probs, dim=1) 
        return torch.cat(predictions, dim=1)


class TransformerTM(GenerativeTM):
    """
    A GenerativeTM that supports Transformer architectures.
    """

    def __init__(self, src_embedder, tgt_embedder, tgt_sos_idx, tgt_eos_idx,
                 encoder, decoder, latent_size, dropout, tied_embeddings,
                 feed_z_method="first"):
        super().__init__()
        self.src_embedder = src_embedder
        self.tgt_embedder = tgt_embedder
        self.tgt_sos_idx = tgt_sos_idx
        self.tgt_eos_idx = tgt_eos_idx
        self.encoder = encoder
        self.decoder = decoder
        self.tied_embeddings = tied_embeddings
        if not self.tied_embeddings:
            self.output_matrix = nn.Parameter(torch.randn(tgt_embedder.num_embeddings, decoder.input_size))
        else:
            self.output_matrix = self.tgt_embedder.weight
        self.dropout_layer = nn.Dropout(p=dropout)

        # For future experiments, different methods of adding z into the transformer.
        assert feed_z_method in ['first', 'none'], "Unknown feed_z_method: {}".format(feed_z_method)
        self.feed_z_method = feed_z_method

        self.fc_z_enc = nn.Linear(latent_size, src_embedder.embedding_dim)
        self.fc_z_dec = nn.Linear(latent_size, tgt_embedder.embedding_dim)

        # A function alias to clean up beam search code.
        self.tgt_embed = self.prepare_decoder_input

    def prepare_decoder_input(self, y, encoder_out, seq_len_x, z):
        """
        embed y, and add z to inputs according to self.feed_z_method.
        """
        y_emb = self.tgt_embedder(y) # [B, T, D]

        if self.feed_z_method == "first":
            # Add z as first decoder input, encoder outputs stay the same.
            z_dec = self.fc_z_dec(z) # [B, D]
            y_emb = torch.cat([z_dec.unsqueeze(1), y_emb], 1) # [B, T+1, D]
        elif self.feed_z_method == "none":
            pass
        else:
            raise NotImplementedError()

        return y_emb, encoder_out, seq_len_x

    def prepare_encoder_input(self, x, seq_len_x, z):
        """
        embed x, and add z to inputs according to self.feed_z_method.
        """
        x_emb = self.src_embedder(x) # [B, T, D]

        if self.feed_z_method == "first":
            # Add z as first encoder input. to both encoder and decoder inputs.
            z_enc = self.fc_z_enc(z) # [B, D]
            x_emb = torch.cat([z_enc.unsqueeze(1), x_emb], 1) # [B, T+1, D]
            # Sequences get 1 longer.
            seq_len_x = seq_len_x.clone() + 1
        elif self.feed_z_method == "none":
            pass
        else:
            raise NotImplementedError()

        return x_emb, seq_len_x

    def encode(self, x, seq_len_x, z):
        x_emb, seq_len_x = self.prepare_encoder_input(x, seq_len_x, z)
        encoder_out, _ = self.encoder(x_emb, seq_len_x)
        return encoder_out, seq_len_x

    def generate(self, pre_output):
        return F.linear(pre_output, self.output_matrix)

    def forward(self, x, seq_mask_x, seq_len_x, y, z, state=dict()) -> Distribution:
        encoder_out, seq_len_x = self.encode(x, seq_len_x, z)

        y_emb, encoder_out, seq_len_x = self.prepare_decoder_input(y, encoder_out, seq_len_x, z)
        decoder_out = self.decoder(y_emb, encoder_out, seq_len_x)
        logits = self.generate(decoder_out)

        if self.feed_z_method == "first":
            # Omit first output, since first input is z instead of start token.
            logits = logits[:, 1:]

        return Categorical(logits=logits)


    def log_prob(self, likelihood: Distribution, y):
        """
        Return log-probability of observation.

        likelihood: as returned by forward
        y: [B, Ty] observed token ids
        """
        return (likelihood.log_prob(y) * (y != self.tgt_embedder.padding_idx).float()).sum(-1)
    
    def sample(self, x, seq_mask_x, seq_len_x, z, max_len=100, greedy=False, state=dict()):
        encoder_out, seq_len_x = self.encoder(x_emb, seq_len_x, z)
        batch_size = x.size(0)
        prev_y = torch.full(size=[batch_size, 1], fill_value=self.tgt_sos_idx, dtype=torch.long, device=x.device)

        # Decode step-by-step. 
        # Because the transformer has no hidden state, the full decoder input is fed at each time step.
        predictions = []
        log_probs = []
        is_complete = torch.zeros_like(prev_y).byte()
        for _ in range(max_len):
            y_emb, encoder_out, seq_len_x = self.prepare_decoder_input(prev_y, encoder_out, seq_len_x, z)
            decoder_out = self.decoder(y_emb, encoder_out, seq_len_x)
            logits = self.generate(decoder_out)
            logit_t = logits[:, -1].unsqueeze(1) #[B, 1, |Y|]
            pyt_xz = Categorical(logits=logit_t)
            if greedy:
                prediction = torch.argmax(logit_t, dim=-1)
            else:
                prediction = pyt_xz.sample()
            prev_y = torch.cat([prev_y, prediction], dim=1)
            log_prob_t = pyt_xz.log_prob(prediction)

            log_probs.append(log_prob_t)
            predictions.append(torch.where(is_complete, torch.full_like(prediction, self.tgt_embedder.padding_idx), prediction))
            is_complete = is_complete | (prediction == self.tgt_eos_idx).byte()

        state['log_prob'] = torch.cat(log_probs, dim=1) 
        return torch.cat(predictions, dim=1)


class TransformerLM(GenerativeLM):
    """
    The forward will return the likelihoods 
        X_i|z,x_{<i}

    To get the likelihood of an observation, use log_prob.
    """

    def __init__(self, embedder, sos_idx, eos_idx, latent_size, hidden_size, 
                 num_heads, num_layers, dropout, tied_embeddings, feed_z_method="first"):
        super().__init__()
        self.embedder = embedder
        self. pad_idx = embedder.padding_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.transformer = TransformerEncoder(
            embedder.embedding_dim, hidden_size, num_heads, num_layers, dropout, autoregressive=True
        )
        self.tied_embeddings = tied_embeddings
        if not self.tied_embeddings:
            self.output_matrix = nn.Parameter(torch.randn(embedder.num_embeddings, self.transformer.input_size))
        else:
            self.output_matrix = self.embedder.weight
        self.dropout_layer = nn.Dropout(p=dropout)

        # For future experiments, different methods of adding z into the transformer.
        assert feed_z_method in ['first', 'none'], "Unknown feed_z_method: {}".format(feed_z_method)
        self.feed_z_method = feed_z_method

        self.fc_z = nn.Linear(latent_size, embedder.embedding_dim)

    
    def prepare_input(self, x, z):
        """
        Embeds the inputs x, and adds in z according to self.feed_z_method.
        """

        x_emb = self.embedder(x) # [B, T, D]
        z_emb = self.fc_z(z) # [B, D]

        if self.feed_z_method == "first":
            # Add z as first input to transformer.
            x_emb = torch.cat([z_emb.unsqueeze(1), x_emb], 1) # [B, T+1, D]
        elif self.feed_z_method == "none":
            pass
        else:
            raise NotImplementedError()

        return x_emb

    def forward(self, x, z, state=dict()) -> Distribution:
        """
        Return distributions X_i|z, x_{<i}.

        x: [B, Tx] token ids
        z: [B, Dz] stochastic embeddings
        state: use this dictionary to return something about the forward pass (for example, 
            attention weights, or other details of the computation that you whish to log)
        """
        x_emb = self.prepare_input(x, z)
        hidden, _ = self.transformer(x_emb)

        if self.feed_z_method == "first":
            hidden = hidden[:, 1:]
        logits = F.linear(hidden, self.output_matrix)

        return Categorical(logits=logits)

    def log_prob(self, likelihood: Distribution, x):
        """
        Return log-probability of observation.

        likelihood: as returned by forward
        x: [B, Tx] observed token ids
        """
        return (likelihood.log_prob(x) * (x != self.pad_idx).float()).sum(-1)

    def sample(self, z, max_len=100, greedy=False, state=dict()):
        """
        Sample from X|z where z [B, Dz]
        """
        batch_size = z.size(0)
        prev_x = torch.full(size=[batch_size, 1], fill_value=self.sos_idx, dtype=torch.long, device=z.device)

        # Decode step-by-step.
        # Because the transformer has no hidden state, the full decoder input is fed at each time step.
        predictions = []
        log_probs = []
        is_complete = torch.zeros_like(prev_x).byte()
        for _ in range(max_len):
            x_emb = self.prepare_input(prev_x, z)
            pre_output, _ = self.transformer(x_emb)
            logits = F.linear(pre_output, self.output_matrix)
            logit_t = logits[:, -1].unsqueeze(1) #[B, 1, |Y|]
            px_z = Categorical(logits=logit_t)
            if greedy:
                prediction = torch.argmax(logit_t, dim=-1)
            else:
                prediction = px_z.sample()
            prev_x = torch.cat([prev_x, prediction], dim=1)
            log_prob_t = px_z.log_prob(prediction)

            log_probs.append(log_prob_t)
            predictions.append(torch.where(is_complete, torch.full_like(prediction, self.embedder.padding_idx), prediction))
            is_complete = is_complete | (prediction == self.eos_idx).byte()

        state['log_prob'] = torch.cat(log_probs, dim=1) 
        return torch.cat(predictions, dim=1)
