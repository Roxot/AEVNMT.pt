"""
Implementation Notes

- TODO Integrate transformer parts into aevnmt model.
- TODO better filenames, components.transformer and models.transformer is confusing.
- TODO Check huggingface for more tricks, might need their optimizers for the lr scheduling etc.
- TODO Is layernorm needed? look at literature, might be detrimental.
- TODO feed_z_method for both the GenerativeLM and GenerativeTM needs testing. Options:
    - z is x_{0} (Current method, feed_z_method="first")
    - x_i -> [z, x_i]: where x_i is the embedding of the ith token in the history
    - h_i -> [z, h_i]: where h_i can be the state just before the output layer,
        and/or the state of the ith token in the memory (attended to)
    - change concatenation to sum in the last two options.
"""

import math

import torch
from torch import nn

def generate_padding_mask(seq_len, max_len=None):
    """
    Generate padding mask for the Transformer Encoder or Decoder memory.

    From the nn.Transformer:
    If a BoolTensor is provided, the positions with the value of True
    will be ignored while the position with the value of False will be unchanged.
    """
    if max_len is None:
        max_len = seq_len.max().int()
    mask = torch.arange(max_len).to(seq_len.device)
    mask = mask.expand(seq_len.size(0), max_len)
    mask = (mask >= seq_len.unsqueeze(1))
    return mask


def generate_square_subsequent_mask(sz):
    """
    Generate a square causal mask. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).

    source: torch.nn.modules.transformer
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class TransformerEncoder(nn.Module):
    """
    An extremely simple wrapper around nn.TransformerEncoder that works in batch major.
    """

    def __init__(self, input_size, hidden_size, num_heads, num_layers, dropout=0., autoregressive=False):
        """
        [summary]

        :param input_size: transformer d_model.
        :param hidden_size: transformer feedforward size.
        :param num_heads: number of attention heads.
        :param num_layers: number of layers.
        :param dropout: dropout chance, defaults to 0.
        :param autoregressive: If True the attention is masked like a transformer decoder, defaults to False
        """
        super().__init__()
        self.pos_enc = PositionalEncoding(input_size, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size,
                                                   nhead=num_heads,
                                                   dim_feedforward=hidden_size,
                                                   dropout=dropout)

        self.transformer_enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_size = input_size
        self.autoregressive = autoregressive

    def forward(self, x_embed, seq_len=None):

        # Transform x_embed from batch major to time major.
        x_embed = x_embed.permute(1, 0, 2) # [T_x, B, input_size]
        x_embed = x_embed * math.sqrt(self.input_size)
        x_embed = self.pos_enc(x_embed)

        # Create a sequence mask.
        if seq_len is not None:
            pad_mask = generate_padding_mask(seq_len, max_len=x_embed.size(0))
        else:
            pad_mask = None

        if self.autoregressive:
            attn_mask = generate_square_subsequent_mask(x_embed.size(0))
        else:
            attn_mask = None


        # Run the transformer encoder.
        x_enc = self.transformer_enc(x_embed, mask=attn_mask, src_key_padding_mask=pad_mask) # [T_x, B, input_size]
        
        # Return in batch major.
        x_enc = x_enc.permute(1, 0, 2) # [B, T_x, input_size]
        x_first = x_enc[:, 0, :]

        return x_enc, x_first # [B, T_x, input_size]


class TransformerDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers, dropout=0.):
        super().__init__()
        self.pos_enc = PositionalEncoding(input_size, dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model=input_size,
                                                   nhead=num_heads,
                                                   dim_feedforward=hidden_size,
                                                   dropout=dropout)

        self.transformer_dec = nn.TransformerDecoder(decoder_layer, num_layers)
        self.input_size = input_size

    def forward(self, y_embed, mem, mem_len):
        # Transform x_embed from batch major to time major.
        y_embed = y_embed.permute(1, 0, 2)
        y_embed = y_embed * math.sqrt(self.input_size)
        y_embed = self.pos_enc(y_embed)

        mem = mem.permute(1, 0, 2)

        # Make padding masks and causal mask.
        mem_pad_mask = generate_padding_mask(mem_len, max_len=mem.size(0))
        causal_mask = generate_square_subsequent_mask(y_embed.size(0))
        out = self.transformer_dec(tgt=y_embed, memory=mem,
                                   tgt_mask=causal_mask,
                                   memory_key_padding_mask=mem_pad_mask)

        return out.permute(1, 0, 2)


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


class TransformerCompositionFunction(nn.Module):
    """
    A composition function to combine (x, y) representations based on a non-causal transformer decoder block.

    Adapted from:
    Latent-Variable Non-Autoregressive Neural Machine Translation with Deterministic Inference Using a Delta Posterior
    (https://github.com/zomux/lanmt/)
    """

    def __init__(self, input_size, hidden_size, num_heads, num_layers, dropout=0.):
        super().__init__()
        layer = nn.TransformerDecoderLayer(d_model=input_size,
                                           nhead=num_heads,
                                           dim_feedforward=hidden_size,
                                           dropout=dropout)
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.input_size = input_size

    def forward(self, x_embed, x_len, y_embed, y_len):
        x_embed = x_embed.permute(1, 0, 2)
        x_pad_mask = generate_padding_mask(x_len, max_len=x_embed.size(0))

        y_embed = y_embed.permute(1, 0, 2)
        y_pad_mask = generate_padding_mask(y_len, max_len=y_embed.size(0))

        out = self.transformer(tgt=x_embed, tgt_key_padding_mask=x_pad_mask,
                               memory=y_embed, memory_key_padding_mask=y_pad_mask)
        out = out.permute(1, 0, 2)
        out_first = out[:, 0]
        return out_first
