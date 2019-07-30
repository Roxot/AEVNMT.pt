import torch
import torch.nn as nn
import torch.nn.functional as F

class LuongAttention(nn.Module):

    def __init__(self, key_size, hidden_size, scale=False):
        """
        (general) from Luong's paper. Multiplicative / bilinear attention.
        """
        super().__init__()
        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.proj_keys = None
        self.key_size = key_size
        self.value_size = key_size
        self.hidden_size = hidden_size
        if scale:
            self.scale = nn.Parameter(torch.tensor([1.0]))
        else:
            self.register_buffer("scale", torch.tensor([1.0]))

    def compute_proj_keys(self, keys):
        """
        :param keys: [B, T_src, H_enc]
        """
        self.proj_keys = self.key_layer(keys) # [B, T_src, hidden_size]

    def forward(self, query, mask, values):
        """
        :param query: [1, B,  H_dec] decoder hidden state
        :param mask: [B, T_src] position mask
        :param values: [B, T_src, H_enc] encoder states
        """
        assert self.proj_keys is not None, "Call compute_proj_keys before computing" \
                                           " the context vectors."
        scores = torch.bmm(query, self.proj_keys.transpose(1, 2)) # [B, 1, T_src]
        scores = scores * self.scale
        scores = torch.where(mask.unsqueeze(1), scores, scores.new_full([1], -float("inf")))
        alphas = F.softmax(scores, dim=-1) # [B, 1, T_src]
        context = torch.bmm(alphas, values)
        return context, alphas

class BahdanauAttention(nn.Module):

    def __init__(self, key_size, query_size, hidden_size):
        super().__init__()
        self.proj_keys = None
        self.key_size = key_size
        self.hidden_size = hidden_size
        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.scores_layer = nn.Linear(hidden_size, 1, bias=False)

    def compute_proj_keys(self, keys):
        """
        :param keys: [B, T_src, H_enc]
        """
        self.proj_keys = self.key_layer(keys) # [B, T_src, hidden_size]

    def forward(self, query, mask, values):
        """
        :param query: [B, 1,  query_size] decoder hidden state
        :param mask: [B, T_src] position mask
        :param values: [B, T_src, H_enc] encoder states
        """
        assert self.proj_keys is not None, "Call compute_proj_keys before computing" \
                                           " the context vectors."

        # Compute the query from the decoder hidden state.
        query = self.query_layer(query) # [B, 1, hidden_size]

        # Compute the attention scores.
        scores = self.scores_layer(torch.tanh(query + self.proj_keys)) # [B, T_src, 1]
        scores = scores.squeeze(-1) # [B, T_src, 1] -> [B, T_src]
        scores = torch.where(mask, scores, scores.new_full([1], -float("inf")))

        # Compute the context vector.
        alphas = F.softmax(scores, dim=-1).unsqueeze(1) # [B, 1, T_src]
        context = torch.bmm(alphas, values)
        return context, alphas
