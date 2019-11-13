"""
Code adapted from https://github.com/bastings/interpretable_predictions/tree/master/latent_rationale/snli
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax(t, mask, dim=-1):
    t = torch.where(mask, t, t.new_full([1], float('-inf')))
    return F.softmax(t, dim=dim)


def get_relative_positions(size, max_dist, device=None):
    """
    Returns the IDs of relative distances, with a max. for bucketing
    e.g. for sentence with 3 words, the relative positions look like:
     0  1  2
    -1  0  1
    -2 -1  0
    to index distance embeddings, we add the maximum distance:
     0+max_dist 1+max_dist, 2+max_dist
    -1+max_dist .. etc.
    etc.
    values larger than max_dist or smallar than -max_dist are clipped
    :param size:
    :param max_dist: maximum relative distance
    :param device: device to create output tensor
    :return: indices for relative distances
    """
    with torch.no_grad():
        v = torch.arange(size, device=device)
        v1 = v.unsqueeze(0)  # broadcast over rows
        v2 = v.unsqueeze(1)  # broadcast over columns
        d = v1 - v2
        d = d.clamp(-max_dist, max_dist) + max_dist
        return d


class AttentionMechanism(nn.Module):

    def __init__(self):
        super(AttentionMechanism, self).__init__()

    def forward(self, *input):
        raise NotImplementedError("Implement this.")


class DotAttention(AttentionMechanism):

    def __init__(self):
        super(DotAttention, self).__init__()

    def forward(self, q, k):
        return q @ k.transpose(1, 2)


class DeepDotAttention(AttentionMechanism):

    def __init__(self, in_features, out_features, dropout=0.2):
        super(DeepDotAttention, self).__init__()

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        self.attention_layer = nn.Sequential(
            nn.Linear(in_features, out_features), self.activation, self.dropout,
            nn.Linear(out_features, out_features), self.activation, self.dropout
        )

    def forward(self, q, k):
        q = self.attention_layer(q)
        k = self.attention_layer(k)
        return q @ k.transpose(1, 2)


class DecomposableAttentionEncoder(nn.Module):
    """
    """

    def __init__(self, x_size, y_size, shared_size, hidden_size, 
            dropout=0.0, mask_diagonal=False, relu_projection=False, use_self_att_dropout=False, max_distance=11):
        """
        x_size: dimensionality of tokens in x 
        y_size: dimensionality of tokens in y 
        shared_size: we project from x_size and y_size to a shared space of dimensionality shared_size 
        hidden_size: dimensionality of the attention head
        """
        super(DecomposableAttentionEncoder, self).__init__()
        
        self.shared_size = shared_size
        self.hidden_size = hidden_size
        
        self.projection_x = nn.Linear(x_size, shared_size)
        self.projection_y = nn.Linear(y_size, shared_size)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()
        
        self.max_dist = max_distance
        self.dist_embed = nn.Embedding(2 * self.max_dist + 1, 1)  # {-max_dist, ..., 0, ... max_dist}
        self.self_attention = DeepDotAttention(
            shared_size, hidden_size, dropout=dropout)

        # set attention mechanism (between premise and hypothesis)
        self.attention = DeepDotAttention(shared_size * 2, hidden_size, dropout=dropout)

        self.compare_layer = nn.Sequential(
            nn.Linear(shared_size * 4, hidden_size), self.activation, self.dropout,
            nn.Linear(hidden_size, hidden_size), self.activation, self.dropout
        )

        self.aggregate_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), self.activation, self.dropout,
            nn.Linear(hidden_size, hidden_size), self.activation, self.dropout
        )

        self.mask_diagonal = mask_diagonal
        self.relu_projection = relu_projection
        self.use_self_att_dropout = use_self_att_dropout

        self.reset_params()

    def reset_params(self):
        """Custom initialization"""

        with torch.no_grad():
            for name, p in self.named_parameters():

                if "embed" in name:
                    continue
                else:
                    if p.dim() > 1:
                        gain = 1.
                        nn.init.xavier_uniform_(p, gain=gain)
                    else:
                        nn.init.zeros_(p)

        if hasattr(self, "dist_embed"):
            std = 1.0
            with torch.no_grad():
                self.dist_embed.weight.normal_(mean=0, std=std)

    def _add_rel_dists(self, x):
        """add matrix of relative distances"""
        rel_dists = get_relative_positions(
            x.size(1), self.max_dist, device=x.device)
        rel_dists = self.dist_embed(rel_dists).squeeze(-1).unsqueeze(0)
        return x + rel_dists

    def _mask_diagonal(self, x):
        """block the diagonal so a word does not self-align"""
        eye = torch.eye(x.size(1), dtype=torch.uint8, device=x.device)
        return torch.where(eye, x.new_full(x.size(), float('-inf')), x)

    def _mask_padding(self, x, mask, value=0.):
        """
        Mask should be true/1 for valid positions, false/0 for invalid ones.
        :param x:
        :param mask:
        :return:
        """
        return torch.where(mask, x, x.new_full([1], value))

    def forward(self, prem, hypo, prem_mask, hypo_mask):
        """
        prem: [B, M, D]
        hypo: [B, N, D]
        """

        #prem_mask = prem_mask.float()
        #hypo_mask = hypo_mask.float()
        # project embeddings to a shared space
        hypo = self.projection_x(hypo)
        prem = self.projection_y(prem)
        if self.relu_projection:
            hypo = self.activation(hypo)
            prem = self.activation(prem)
        # [B, M, D]
        hypo = self.dropout(hypo)
        # [B, N, D]
        prem = self.dropout(prem)

        # this is the original self-attention from DA

        # self-attention (self dot product)
        # [B, M, M]
        prem_self_att = self.self_attention(prem, prem)
        # [B, N, N]
        hypo_self_att = self.self_attention(hypo, hypo)

        # add relative distances
        prem_self_att = self._add_rel_dists(prem_self_att)
        hypo_self_att = self._add_rel_dists(hypo_self_att)

        if self.mask_diagonal:
            prem_self_att = self._mask_diagonal(prem_self_att)
            hypo_self_att = self._mask_diagonal(hypo_self_att)

        prem_self_att = masked_softmax(prem_self_att, prem_mask.unsqueeze(1))
        hypo_self_att = masked_softmax(hypo_self_att, hypo_mask.unsqueeze(1))

        # [B, M, D]
        prem_self_att_ctx = prem_self_att @ prem
        # [B, N, D]
        hypo_self_att_ctx = hypo_self_att @ hypo

        if self.use_self_att_dropout:
            prem_self_att_ctx = self.dropout(prem_self_att_ctx)
            hypo_self_att_ctx = self.dropout(hypo_self_att_ctx)
        
        # [B, M, 2D]
        prem = torch.cat([prem, prem_self_att_ctx], dim=-1)
        # [B, N, 2D]
        hypo = torch.cat([hypo, hypo_self_att_ctx], dim=-1)

        # compute attention
        # [B, M, N]
        sim = self.attention(prem, hypo)

        # [B, M, N]
        prem2hypo_att = masked_softmax(sim, hypo_mask.unsqueeze(1))
        # [B, N, M]
        hypo2prem_att = masked_softmax(
            sim.transpose(1, 2).contiguous(), prem_mask.unsqueeze(1))

        # take weighed sum of hypo (premise) based on attention weights
        # [B, M, 2D]
        attended_hypo = prem2hypo_att @ hypo
        # [B, N, 2D]
        attended_prem = hypo2prem_att @ prem

        # compare input
        prem_compared = self.compare_layer(
            torch.cat([prem, attended_hypo], dim=-1))
        hypo_compared = self.compare_layer(
            torch.cat([hypo, attended_prem], dim=-1))

        prem_compared = prem_compared * prem_mask.float().unsqueeze(-1)
        hypo_compared = hypo_compared * hypo_mask.float().unsqueeze(-1)

        prem_compared = prem_compared.sum(dim=1)
        hypo_compared = hypo_compared.sum(dim=1)

        aggregate = self.aggregate_layer(
            torch.cat([prem_compared, hypo_compared], dim=-1))

        return aggregate


