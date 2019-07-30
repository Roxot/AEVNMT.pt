import torch.nn as nn

def rnn_creation_fn(cell_type):
    cell_type = cell_type.lower()
    if cell_type == "rnn":
        return nn.RNN
    elif cell_type == "gru":
        return nn.GRU
    elif cell_type == "lstm":
        return nn.LSTM

def tile_rnn_hidden(hidden, rnn):
    """
    :param hidden: [1, B, hidden_size]
    :param rnn: an instance of torch.nn.RNN, torch.nn.GRU. or torch.nn.LSTM.

    Returns a tiled hidden state that can serve as input to the given rnn. Takes into account
    cell type, bidirectionality, and number of layers.
    """
    num_layers = rnn.num_layers
    num_layers = num_layers * 2 if rnn.bidirectional else num_layers
    hidden = hidden.repeat(num_layers, 1, 1)
    if isinstance(rnn, nn.LSTM):
        hidden = (hidden, hidden)
    return hidden
