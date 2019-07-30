import math
import torch
import torch.nn as nn

from torch.nn.init import _calculate_fan_in_and_fan_out

def xavier_uniform_n_(w, gain=1., n=4):
    """
    From: https://github.com/joeynmt/joeynmt/blob/master/joeynmt/initialization.py

    Xavier initializer for parameters that combine multiple matrices in one
    parameter for efficiency. This is e.g. used for GRU and LSTM parameters,
    where e.g. all gates are computed at the same time by 1 big matrix.
    :param w: parameter
    :param gain: default 1
    :param n: default 4
    """
    with torch.no_grad():
        fan_in, fan_out = _calculate_fan_in_and_fan_out(w)
        assert fan_out % n == 0, "fan_out should be divisible by n"
        fan_out //= n
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        nn.init.uniform_(w, -a, a)

def initialize_model(model, pad_idx, cell_type, emb_init_scale, verbose=False):
    """
    Initializes the model parameters. Makes use of naming conventions for parameters
    that all models in aevnmt.models should follow.
    """

    xavier_gain = 1.

    with torch.no_grad():
        for name, param in model.named_parameters():

            # Initialize embeddings from a 0-mean Gaussian with scale emb_init_scale.
            if "embedder" in name:
                if verbose:
                    print(f"Initializing {name} with N(0, {emb_init_scale})")
                nn.init.normal_(param, mean=0., std=emb_init_scale)

                # Set the padding embedding back to zeros.
                param[pad_idx].zero_()

            # Initialize biases to zeros.
            elif "bias" in name:
                if verbose:
                    print(f"Initializing {name} to 0")
                nn.init.zeros_(param)

            # Initialize RNN weight matrices with Xavier uniform initialization.
            elif "rnn." in name:
                if verbose:
                    print(f"Initializing {name} with xavier_uniform(gain={xavier_gain:.1f})"
                          f" for {cell_type}")
                n = 4 if cell_type == "lstm" else 3
                xavier_uniform_n_(param.data, gain=xavier_gain, n=n)

            # For all other matrices just use Xavier uniform initialization.
            elif len(param) > 1:
                if verbose:
                    print(f"Initializing {name} with xavier_uniform(gain={xavier_gain:.1f})")
                nn.init.xavier_uniform_(param, gain=xavier_gain)
