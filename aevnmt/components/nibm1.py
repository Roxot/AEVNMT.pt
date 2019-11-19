"""
This implements a neural version of an IBM1 likelihood, that is, 

    P(y|x) = \prod_{j=1}^n \sum_{i=1}^m P(a_j=i)P(y_j|x_i)

where n = |y|, m = |x|, P(a_j) = 1/m, and P(y_j|x_i) = Cat(y_j|NN(x_i)).
"""
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np


class NeuralIBM1(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size, input_size, hidden_size, pad_idx):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.pad_idx = pad_idx
        self.translation_layer = nn.Sequential(nn.Linear(input_size, hidden_size),
                                               nn.ReLU(),
                                               nn.Linear(hidden_size, tgt_vocab_size),
                                               nn.Softmax(dim=-1))

    def forward(self, x, seq_mask_x, seq_len_x, longest_y):
        """
        Return a batch of marginal probabilities for distributions Y_j | x with shape [B, Ty, V].

        x: [B, Tx, Dx]
        seq_mask_x: [B, Tx]
        seq_len_x: [B]
        logest_y: Ty
        """
        # Compute p(y_j|x_i, a_j) for all x_i in x.
        batch_size = x.size(0)
        longest_x = x.size(1)
        # [B * Tx, Dx]
        py_given_xa = self.translation_layer(x.view(batch_size * longest_x, self.input_size))
        # [B, Tx, Vy]
        py_given_xa = py_given_xa.view(batch_size, longest_x, self.tgt_vocab_size)

        # P(a_1^T_y|l, m) = 1 / (T_x + 1) -- note that the NULL word is added to x,
        # seq_mask_x and seq_len_x already.
        # [B, Tx]
        p_align = seq_mask_x.type_as(x) / seq_len_x.unsqueeze(-1).type_as(x)

        # Tile p_align to [B, Ty, Tx].
        # [B, 1, Tx]
        p_align = p_align.unsqueeze(1)
        # [B, Ty, Tx]
        p_align = p_align.repeat(1, longest_y, 1)

        # Compute the marginal p(y|x)
        # [B, Ty, Vy]
        p_marginal = torch.bmm(p_align, py_given_xa)

        return p_marginal
    
    def loss(self, p_marginal, y):
        """
        Returns marginal log-likelihood of observations p(y|x) with shape [B]

        p_marginal: [B, Ty, Vy]
        y: [B, Ty]
        """
        # [B, Ty, 1]
        p_observed = torch.gather(p_marginal, -1, y.unsqueeze(-1))
        # [B, Ty]
        p_observed = p_observed.squeeze(-1)
        # [B]
        log_likelihood = torch.log(p_observed).sum(dim=1)
        return -log_likelihood
    
    def log_likelihood(self, x, seq_mask_x, seq_len_x, y):
        """
        Returns marginal log-likelihood of observations p(y|x) with shape [B]

        x: [B, Tx, Dx]
        seq_mask_x: [B, Tx]
        seq_len_x: [B]
        y: [B, Ty]
        """
        # [B, Ty, Vy]
        p_marginal = self(x, seq_mask_x, seq_len_x, y)
        # [B, Ty, 1]
        p_observed = torch.gather(p_marginal, -1, y.unsqueeze(-1))
        # [B, Ty]
        p_observed = p_observed.squeeze(-1)
        # [B]
        log_likelihood = torch.log(p_observed).sum(dim=1)
        return log_likelihood

    def align(self, x, seq_mask_x, seq_len_x, y):
        """
        Returns a batch of marginal distributions Y_j | x with shape [B, Ty, V].
        

        x: [B, Tx, Dx]
        seq_mask_x: [B, Tx]
        seq_len_x: [B]
        y: [B, Ty]
        """

        with torch.no_grad():
            # Compute P(y|x, a)
            batch_size = x.size(0)
            longest_x = x.size(1)
            py_given_xa = self.translation_layer(x.view(batch_size * longest_x, self.input_size))
            py_given_xa = py_given_xa.view(batch_size, longest_x, self.tgt_vocab_size) # [B, T_x, V_y]
            py_given_xa = torch.where(seq_mask_x.unsqueeze(-1), py_given_xa,
                                            py_given_xa.new_full([1], -float("inf")))

            longest_y = y.size(1)
            alignments = np.zeros([batch_size, longest_y], dtype=np.int)

            # Take the argmax_a P(y|x, a) for each y_j. Note that we can do this as the
            # alignment probabilities are constant and the alignments are independent.
            # I.e., this is identical to argmax_a P(a|x, f).
            for batch_idx, y_n in enumerate(y):
                for j, y_j in enumerate(y_n):
                    if y_j == self.pad_idx:
                        break
                    p = py_given_xa[batch_idx, :, y_j]
                    alignments[batch_idx, j] = np.argmax(p.cpu())

        return alignments

