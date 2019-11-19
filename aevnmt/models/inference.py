"""
Inference models

* InferenceEncoder: maps from sequence pairs (x,y) to a fixed-dimension vector. Some of the encoders available ignore y (they can be used as prediction models).
* InferenceModel: encapsulates an InferenceEncoder to encode (x) or (x, y) and a Conditioner to parameterise a distribution of choice.
"""
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F

from aevnmt.dist import NormalLayer, KumaraswamyLayer
from aevnmt.components import RNNEncoder, TransformerEncoder
from aevnmt.components import DecomposableAttentionEncoder


class InferenceEncoder(nn.Module):
    """
    The encoder of AEVNMT's inference network can one or two sequences, depending on whether we want to model
        q(z|x) or q(z|x,y).
    This is a general interface, it exposes two properties
        - hidden_size
        - output_size
    and fixes a signature for the forward method.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y):
        """
        x: [B, M]
        seq_mask_x: [B, M]
        seq_len_x: [B]
        y: [B, N]
        seq_mask_y: [B, N]
        seq_len_y: [B]

        Returns [B, D]
        """
        raise NotImplementedError("Implement me!")


class RecurrentEncoderX(InferenceEncoder):
    """
    Encodes a sequence (e.g. x) into a fixed-dimension vector using a recurrent cell (possibly bidirectional).
    """

    def __init__(self, embedder, hidden_size, num_layers, cell_type, bidirectional=True, composition="avg", dropout=0.):
        super().__init__()
        self.embedder = embedder
        self.rnn = RNNEncoder(
            emb_size=embedder.embedding_dim,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            dropout=dropout,
            num_layers=num_layers,
            cell_type=cell_type)
        self.hidden_size = hidden_size
        self.output_size = hidden_size if not bidirectional else hidden_size * 2
        self.composition = composition
        if composition not in ["avg","maxpool"]:
            raise NotImplementedError("I only support average and maxpool, but I welcome contributions!")

    def forward(self, x, seq_mask_x, seq_len_x, y=None, seq_mask_y=None, seq_len_y=None):
        x_embed = self.embedder(x).detach()
        encoder_outputs, _ = self.rnn(x_embed, seq_len_x)
        if self.composition == "maxpool":
            avg_encoder_output = encoder_outputs.max(dim=1)[0]
        else:
            avg_encoder_output = (encoder_outputs * seq_mask_x.unsqueeze(-1).type_as(encoder_outputs)).sum(dim=1)
        return avg_encoder_output


class TransformerEncoderX(InferenceEncoder):
    """
    Encodes a sequence (e.g. x) into a fixed-dimension vector using a recurrent cell (possibly bidirectional).
    """

    def __init__(self, embedder, hidden_size, num_heads, num_layers, dropout=0.):
        super().__init__()
        self.embedder = embedder
        self.transformer = TransformerEncoder(
            input_size=embedder.embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_ff=hidden_size,
            dropout=dropout)
        self.hidden_size = hidden_size
        self.output_size = hidden_size

    def forward(self, x, seq_mask_x, seq_len_x, y=None, seq_mask_y=None, seq_len_y=None):
        x_embed = self.embedder(x).detach()
        encoder_outputs, first = self.transformer(x_embed, seq_len_x)
        # TODO: support max pooling (and perhaps other composition functions)
        return first


class NLIEncoderXY(InferenceEncoder):
    """
    Encodes a pair of sequences (i.e. x, y) into a fixed-dimension vector using NLI's Decomposable Attention Model.
    """

    def __init__(self, embedder_x, embedder_y, shared_size, hidden_size,
            max_distance=20, dropout=0., mask_diagonal=False, relu_projection=False, use_self_att_dropout=False):
        super().__init__()
        self.embedder_x = embedder_x
        self.embedder_y = embedder_y
        self.nli = DecomposableAttentionEncoder(
            x_size=embedder_x.embedding_dim,
            y_size=embedder_y.embedding_dim,
            shared_size=shared_size,
            hidden_size=hidden_size,
            max_distance=max_distance,
            dropout=dropout,
            mask_diagonal=mask_diagonal,
            relu_projection=relu_projection,
            use_self_att_dropout=use_self_att_dropout)
        self.hidden_size = hidden_size
        self.output_size = hidden_size

    def forward(self, x, seq_mask_x, seq_len_x, y=None, seq_mask_y=None, seq_len_y=None):
        x_embed = self.embedder_x(x).detach()
        y_embed = self.embedder_y(y).detach()
        outputs = self.nli(prem=x_embed, hypo=y_embed, prem_mask=seq_mask_x, hypo_mask=seq_mask_y)
        return outputs


def get_inference_encoder(encoder_style: str, conditioning_context: str,
        embedder_x, embedder_y, hidden_size,
        rnn_bidirectional, rnn_num_layers, rnn_cell_type,
        transformer_heads, transformer_layers,
        nli_shared_size, nli_max_distance,
        dropout=0.0,composition="avg") -> InferenceEncoder:
    """Creates the appropriate encoder as a function of encoder_style and conditioning_context."""
    if encoder_style == "rnn":
        if conditioning_context == "x":
            encoder = RecurrentEncoderX(
                embedder=embedder_x,
                hidden_size=hidden_size,
                bidirectional=rnn_bidirectional,
                num_layers=rnn_num_layers,
                cell_type=rnn_cell_type,
                dropout=dropout,composition=composition)
        else:
            raise NotImplementedError("I cannot yet condition on the pair (x,y) with an RNN, but I welcome contributions!")
    elif encoder_style == "transformer":
        if conditioning_context == "x":
            encoder = TransformerEncoderX(
                embedder=embedder_x,
                hidden_size=hidden_size,
                num_heads=transformer_heads,
                num_layers=transformer_layers,
                dropout=dropout)
        else:
            raise NotImplementedError("I cannot yet condition on the pair (x,y) with a Transformer, but I welcome contributions!")
    elif encoder_style == "nli":
        if conditioning_context == "xy":
            encoder = NLIEncoderXY(
                embedder_x=embedder_x,
                embedder_y=embedder_y,
                shared_size=nli_shared_size,
                hidden_size=hidden_size,
                max_distance=nli_max_distance,
                dropout=dropout,
                mask_diagonal=False,
                relu_projection=False,
                use_self_att_dropout=False)
        else:
            raise NotImplementedError("NLI encoder for the inference model assumes you are conditioning on the pair (x,y).")
    else:
        raise NotImplementedError("I do not yet support encoder_style=%r" % encoder_style)
    return encoder

class InferenceModel(nn.Module):

    def __init__(self, family: str, latent_size: int, hidden_size: int, encoder: InferenceEncoder):
        """
        :param src_embedder: uses this embedder, but detaches its output from the graph as to not compute
                             gradients for it.
        """
        super().__init__()
        self.family = family
        self.encoder = encoder
        if family == "gaussian":
            self.conditioner = NormalLayer(encoder.output_size, hidden_size, latent_size)
        elif family == "kumaraswamy":
            self.conditioner = KumaraswamyLayer(encoder.output_size, hidden_size, latent_size)
        else:
            raise NotImplementedError("I cannot design %s posterior approximation." % family)

    def forward(self, x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y) -> torch.distributions.Distribution:
        # [B, D]
        outputs = self.encoder(x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y)
        return self.conditioner(outputs)

    def parameters(self, recurse=True):
        return chain(self.encoder.parameters(recurse=recurse), self.conditioner.parameters(recurse=recurse))

    def named_parameters(self, prefix='', recurse=True):
        return chain(self.encoder.named_parameters(prefix='', recurse=True), self.conditioner.named_parameters(prefix='', recurse=True), )
