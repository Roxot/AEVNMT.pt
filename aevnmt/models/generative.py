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
from .nibm1 import NeuralIBM1

from dgm.conditional import MADEConditioner
from dgm.likelihood import AutoregressiveLikelihood


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
        raise NotImplementedError("Implement me!")

    def log_prob(self, likelihood: Distribution, x):
        raise NotImplementedError("Implement me!")


class IndependentLM(GenerativeLM):

    def __init__(self, latent_size, src_vocab_size, pad_idx):
        super().__init__()
        self.pad_idx = pad_idx
        self.output_layer = nn.Linear(latent_size, src_vocab_size, bias=True)
    
    def forward(self, x, z) -> Categorical:
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

class MADELM(nn.Module):

    def __init__(self, vocab_size, latent_size, hidden_sizes, pad_idx, num_masks=10):
        super().__init__()
        self.pad_idx = pad_idx
        self.made_conditioner = MADEConditioner(
            input_size=vocab_size + latent_size, 
            output_size=vocab_size,  
            context_size=latent_size,
            hidden_sizes=hidden_sizes,
            num_masks=num_masks
        )
        self.made = AutoregressiveLikelihood(
            event_size=vocab_size, 
            dist_type=Bernoulli,
            conditioner=self.made_conditioner
        )

    def forward(self, x, z):
        bsz = x.size(0)
        seq_mask_x = x != self.pad_idx
        made_input = torch.zeros((bsz,self.made.event_size),device=x.device)
        for i in range(bsz):
            bow = torch.unique(x[i] * seq_mask_x[i].type_as(x[i]))
            made_input[i][bow] = 1.0
        return self.made(z, made_input)

    def log_prob(self, likelihood, x):
        bsz = x.size(0)
        made_ref = torch.zeros((bsz,self.made.event_size),device=x.device)
        for i in range(bsz):
            bow = torch.unique(x[i])
            made_ref[i][bow] = 1.0
        return likelihood.log_prob(made_ref).sum(-1)


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
        raise NotImplementedError("Implement me!")

    def log_prob(self, likelihood: Distribution, y):
        raise NotImplementedError("Implement me!")


class IndependentTM(GenerativeTM):

    def __init__(self, latent_size, tgt_vocab_size, pad_idx):
        super().__init__()
        self.pad_idx = pad_idx
        self.output_layer = nn.Linear(latent_size, tgt_vocab_size, bias=True)
    
    def forward(self, x, seq_mask_x, seq_len_x, y, z) -> Categorical:
        # [B, 1, Vy]
        return Categorical(logits=self.output_layer(z).unsqueeze(1))
    
    def log_prob(self, likelihood: Categorical, y):
        # [B, Ty, Vy]
        likelihood = likelihood.expand(y.size())
        # [B, Ty]
        log_p = likelihood.log_prob(y)
        # mask invalid positions
        log_p = log_p * (y != self.pad_idx).float()
        # [B]
        log_p = log_p.sum(-1)
        return log_p

class IBM1TM(GenerativeTM):

    def __init__(self, src_embed, latent_size, hidden_size, src_vocab_size, tgt_vocab_size, pad_idx):
        super().__init__()
        self.pad_idx = pad_idx
        self.src_embed = src_embed
        self.gate = nn.Sequential(
            nn.Linear(src_embed.embedding_dim, latent_size), 
            nn.Sigmoid()
        )
        self.proj = nn.Linear(latent_size, latent_size)
        self.nibm1 = NeuralIBM1(src_vocab_size, tgt_vocab_size, latent_size, hidden_size, pad_idx)

    def forward(self, x, seq_mask_x, seq_len_x, y, z) -> Categorical:
        # [B, Tx, Dx]
        x = self.src_embed(x)
        # [B, Tx, Dz]
        g = self.gate(x)
        # [B, Tx, Dz]
        x = self.proj(g * z.unsqueeze(1))
        # [B, Ty, V]
        marginals = - self.nibm1(x, seq_mask_x, seq_len_x, y.size(1))
        return Categorical(probs=marginals)

    def log_prob(self, likelihood: Categorical, y):
        return (likelihood.log_prob(y) * (y != self.pad_idx).float()).sum(-1)

