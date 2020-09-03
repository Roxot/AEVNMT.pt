"""
Inference models

* InferenceEncoder: maps from sequence pairs (x,y) to a fixed-dimension vector. Some of the encoders available ignore y (they can be used as prediction models).
* InferenceModel: 
    * BasicInferenceModel: encapsulates an InferenceEncoder to encode x, y, or (x, y) and a Conditioner to parameterise a distribution of choice.
    * SwitchingInferenceModel: encapsulates 3 InferenceModels, one conditioned on x, one conditioned on y, one conditioned on (x,y)
"""
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution

from aevnmt.dist import NormalLayer, KumaraswamyLayer
from aevnmt.dist import Conditioner
from aevnmt.components import RNNEncoder, DecomposableAttentionEncoder
from aevnmt.components import TransformerEncoder, TransformerDecoder, TransformerCompositionFunction


class InferenceEncoder(nn.Module):
    """
    The encoder of AEVNMT's inference network can one or two sequences, depending on whether we want to model
        q(z|x) or q(z|x,y).
    This is a general interface, it exposes two properties
        - output_size
    and fixes a signature for the forward method, where we always expect the pair (x,y). Though note,
    not every implementation will necessarily use x and y.
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
        self.output_size = hidden_size if not bidirectional else hidden_size * 2
        self.hidden_size = hidden_size
        self.composition = composition
        if composition not in ["avg","maxpool"]:
            raise NotImplementedError("I only support average and maxpool, but I welcome contributions!")

    def forward(self, x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y):
        x_embed = self.embedder(x).detach()
        encoder_outputs, _ = self.rnn(x_embed, seq_len_x)
        if self.composition == "maxpool":
            avg_encoder_output = encoder_outputs.max(dim=1)[0]
        else:
            avg_encoder_output = (encoder_outputs * seq_mask_x.unsqueeze(-1).type_as(encoder_outputs)).sum(dim=1)
        return avg_encoder_output


class RecurrentEncoderY(InferenceEncoder):
    """
    Encodes a sequence (e.g. x) into a fixed-dimension vector using a recurrent cell (possibly bidirectional).
    """

    def __init__(self, embedder, hidden_size, num_layers, cell_type, bidirectional=True, composition="avg", dropout=0.):
        super().__init__()
        self.encoder = RecurrentEncoderX(embedder, hidden_size, num_layers, cell_type, bidirectional, composition, dropout)

    @property
    def output_size(self):
        return self.encoder.output_size

    def forward(self, x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y):
        return self.encoder(y, seq_mask_y, seq_len_y, None, None, None)


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

    def forward(self, x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y):
        x_embed = self.embedder_x(x).detach()
        y_embed = self.embedder_y(y).detach()
        outputs = self.nli(prem=x_embed, hypo=y_embed, prem_mask=seq_mask_x, hypo_mask=seq_mask_y)
        return outputs


class CombinedEncoder(InferenceEncoder):

    def __init__(self, encoder_x: InferenceEncoder, encoder_y: InferenceEncoder, composition='cat'):
        assert composition in ["cat"], "I do not know this composition function: '%s'" % composition
        super().__init__()
        self.encoder_x = encoder_x
        self.encoder_y = encoder_y
        self.composition = composition

    @property
    def output_size(self):
        if self.composition == 'cat':
            return self.encoder_x.output_size + self.encoder_y.output_size
        else:
            NotImplementedError("I do not know what to do")

    def forward(self, x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y):
        hx = self.encoder_x(x, seq_mask_x, seq_len_x, None, None, None)
        hy = self.encoder_y(None, None, None, y, seq_mask_y, seq_len_y)
        # TODO: implement composition
        return torch.cat([hx, hy], -1)


class TransformerEncoderX(InferenceEncoder):
    """
    Encodes a sequence (e.g. x) using a transformer encoder. To obtain a fixed dimension vector instead of the
    full transformer output, use composition="first".
    """

    def __init__(self, embedder, hidden_size, num_heads, num_layers, dropout=0., composition="none"):
        super().__init__()
        self.embedder = embedder
        self.transformer = TransformerEncoder(
            input_size=embedder.embedding_dim,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout)
        self.hidden_size = hidden_size
        self.output_size = hidden_size
        self.composition = composition

    def forward(self, x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y):
        x_embed = self.embedder(x).detach()
        encoder_outputs, first = self.transformer(x_embed, seq_len_x)

        if self.composition == "none":
            return encoder_outputs
        elif self.composition == "first":
            return first
        else:
            raise RuntimeError("Unknown composition method: {}".format(self.composition))


class TransformerEncoderY(InferenceEncoder):
    """
    Encodes a sequence (e.g. x) into a fixed-dimension vector using a recurrent cell (possibly bidirectional).
    """
    
    def __init__(self, embedder, hidden_size, num_heads, num_layers, dropout=0., composition="none"):
        super().__init__()
        self.encoder = TransformerEncoderX(embedder, hidden_size, num_heads, num_layers, dropout, composition=composition)

    @property
    def output_size(self):
        return self.encoder.output_size

    def forward(self, x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y):
        return self.encoder(y, seq_mask_y, seq_len_y, None, None, None)


class CombinedTransformerEncoder(InferenceEncoder):

    def __init__(self, encoder_x: InferenceEncoder, encoder_y: InferenceEncoder, composition):
        super().__init__()
        self.encoder_x = encoder_x
        self.encoder_y = encoder_y
        self.composition = composition

    @property
    def output_size(self):
        raise NotImplementedError("I do not know what to do")

    def forward(self, x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y):
        hx = self.encoder_x(x, seq_mask_x, seq_len_x, None, None, None)
        hy = self.encoder_y(None, None, None, y, seq_mask_y, seq_len_y)
        comp = self.composition(hx, seq_len_x, hy, seq_len_y)
        return comp
        

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
        elif conditioning_context == "y":
            encoder = RecurrentEncoderY(
                embedder=embedder_y,
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
                dropout=dropout,
                composition="first")
        elif conditioning_context == "y":
            encoder = TransformerEncoderY(
                embedder=embedder_y,
                hidden_size=hidden_size,
                num_heads=transformer_heads,
                num_layers=transformer_layers,
                dropout=dropout,
                composition="first")
        else:
            # TODO add CombinedTransformerEncoder and test. Implementation already exists in models.transformer
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


def combine_inference_encoders(encoder_x: InferenceEncoder, encoder_y: InferenceEncoder, composition: str) -> InferenceEncoder:
    return CombinedEncoder(encoder_x, encoder_y, composition)
    

class InferenceModel(nn.Module):

    def __init__(self, latent_size):
        super().__init__()
        self._latent_size = latent_size

    @property
    def latent_size(self):
        return self._latent_size

    def forward(self, x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y) -> Distribution:
        raise NotImplementedError("Implement me!")

    def parameters(self, recurse=True):
        return NotImplementedError("Implement me!")

    def named_parameters(self, prefix='', recurse=True):
        return NotImplementedError("Implement me!")


class BasicInferenceModel(InferenceModel):
    """
    Encodes the inputs using an encoder and then parameterises a variational family.
    """

    def __init__(self, latent_size: int, conditioner: Conditioner,  encoder: InferenceEncoder):
        """
        :param src_embedder: uses this embedder, but detaches its output from the graph as to not compute
                             gradients for it.
        """
        super().__init__(latent_size)
        self.encoder = encoder
        self.conditioner = conditioner

    def forward(self, x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y) -> Distribution:
        # [B, D]
        outputs = self.encoder(x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y)
        return self.conditioner(outputs)

    def parameters(self, recurse=True):
        return chain(self.encoder.parameters(recurse=recurse), self.conditioner.parameters(recurse=recurse))

    def named_parameters(self, prefix='', recurse=True):
        return chain(self.encoder.named_parameters(prefix='', recurse=True), self.conditioner.named_parameters(prefix='', recurse=True), )


class SwitchingInferenceModel(InferenceModel):
    """
    Combines 3 types of inference models: q(z|x), q(z|y), and q(z|x,y).
    """

    def __init__(self, model_x: InferenceModel, model_y: InferenceModel, model_xy: InferenceModel):
        assert model_x.latent_size == model_y.latent_size == model_xy.latent_size, 'Different latent sizes'
        super().__init__(model_x.latent_size)
        self.model_x = model_x
        self.model_y = model_y
        self.model_xy = model_xy

    def forward(self, x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y) -> Distribution:
        if x is not None and y is not None:
            return self.model_xy(x, seq_mask_x, seq_len_x, y, seq_mask_y, seq_len_y)
        elif x is not None and y is None:
            return self.model_x(x, seq_mask_x, seq_len_x, None, None, None)
        elif y is not None and x is None:
            return self.model_y(None, None, None, y, seq_mask_y, seq_len_y)
        else:
            raise ValueError('I cannot perform inferences from nothing')

    def parameters(self, recurse=True):
        return chain(self.model_x.parameters(recurse=recurse), self.model_y.parameters(recurse=recurse), self.model_xy.parameters(recurse=recurse))

    def named_parameters(self, prefix='', recurse=True):
        return chain(self.model_x.named_parameters(prefix='conditioning_x', recurse=True), self.model_y.named_parameters(prefix='conditioning_y', recurse=True), self.model_xy.named_parameters(prefix='conditioning_xy', recurse=True))



