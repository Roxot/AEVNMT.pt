import torch
import torch.nn as nn
import math

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .utils import rnn_creation_fn
from .nli import DecomposableAttentionEncoder as NLIEncoder

class RNNEncoder(nn.Module):

    def __init__(self, emb_size, hidden_size, bidirectional=False,
                 dropout=0., num_layers=1, cell_type="lstm"):
        super().__init__()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.cell_type = cell_type
        self.hidden_size = hidden_size
        rnn_dropout = 0. if num_layers == 1 else dropout
        rnn_fn = rnn_creation_fn(cell_type)
        self.rnn = rnn_fn(emb_size, hidden_size, batch_first=True,
                          bidirectional=bidirectional, dropout=rnn_dropout,
                          num_layers=num_layers)

    def forward(self, x_embed, seq_len, hidden=None):
        """
        Assumes x is sorted by length in desc. order.
        """

        # Run the RNN over the entire sentence.
        packed_seq = pack_padded_sequence(x_embed, seq_len, batch_first=True, enforce_sorted=False)
        output, final = self.rnn(packed_seq, hidden)
        output, _ = pad_packed_sequence(output, batch_first=True)

        # Take h as final state for an LSTM.
        if self.cell_type == "lstm":
            final = final[0] # h out of (h, c) [layers, B, hidden_size]

        # Concatenate the final states of each layer.
        layers = [final[layer_num] for layer_num in range(final.size(0))]
        final_combined = torch.cat(layers, dim=-1) # [B, layers * hidden_size]

        return output, final_combined

class TransformerEncoder(nn.Module):
    """
    An extremely simple wrapper around nn.TransformerEncoder that works in batch major.
    """

    def __init__(self, input_size, num_heads, num_layers, dim_ff, dropout=0.):
        super().__init__()
        self.pos_enc = PositionalEncoding(input_size, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size,
                                                   nhead=num_heads,
                                                   dim_feedforward=dim_ff,
                                                   dropout=dropout)
                                                   
        self.transformer_enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_size = input_size

    def forward(self, x_embed, seq_len):
                
        # Transform x_embed from batch major to time major.
        x_embed = x_embed.permute(1, 0, 2) # [T_x, B, input_size]
        x_embed = x_embed * math.sqrt(self.input_size)
        x_embed = self.pos_enc(x_embed)

        # Create a sequence mask.
        max_len = seq_len.max()
        src_mask = torch.arange(max_len).to(seq_len.device)
        src_mask = src_mask.expand(seq_len.size(0), max_len.int())
        src_mask = (src_mask >= seq_len.unsqueeze(1))
   
        # Run the transformer encoder.
        x_enc = self.transformer_enc(x_embed, src_key_padding_mask=src_mask) # [T_x, B, input_size]
        # Return in batch major.
        x_enc = x_enc.permute(1, 0, 2) # [B, T_x, input_size]
        x_first = x_enc[:, 0, :]

        return x_enc, x_first # [B, T_x, input_size]

class PositionalEncoding(nn.Module):
    """
    From: https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    """

    def __init__(self, d_model, dropout=0.1, max_len=250):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

