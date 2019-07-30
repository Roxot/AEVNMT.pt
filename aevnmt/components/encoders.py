import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .utils import rnn_creation_fn

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
