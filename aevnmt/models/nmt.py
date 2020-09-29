import torch
import torch.nn as nn
import torch.nn.functional as F
from aevnmt.components import label_smoothing_loss
from torch.distributions import Categorical


class ConditionalNMT(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size, emb_size, encoder, decoder,
                 pad_idx, dropout, tied_embeddings,feed_z_size=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedder = nn.Embedding(src_vocab_size, emb_size, padding_idx=pad_idx)
        self.tgt_embedder = nn.Embedding(tgt_vocab_size, emb_size, padding_idx=pad_idx)
        self.tied_embeddings = tied_embeddings
        if not tied_embeddings:
            self.output_matrix = nn.Parameter(torch.randn(tgt_vocab_size, decoder.hidden_size))
        self.dropout_layer = nn.Dropout(p=dropout)

    def generative_parameters(self):
        return self.parameters()

    def inference_parameters(self):
        return None

    def lagrangian_parameters(self):
        return None

    def src_embed(self, x):
        x_embed = self.src_embedder(x)
        x_embed = self.dropout_layer(x_embed)
        return x_embed

    def tgt_embed(self, y):
        y_embed = self.tgt_embedder(y)
        y_embed = self.dropout_layer(y_embed)
        return y_embed

    def encode(self, x, seq_len_x):
        x_embed = self.src_embed(x)
        return self.encoder(x_embed, seq_len_x)

    def init_decoder(self, encoder_outputs, encoder_final):
        hidden = self.decoder.init_decoder(encoder_outputs, encoder_final)
        return hidden

    def generate(self, pre_output):
        W = self.tgt_embedder.weight if self.tied_embeddings else self.output_matrix
        return F.linear(pre_output, W)

    def forward(self, x, seq_mask_x, seq_len_x, y):

        # Encode the source sentence and initialize the decoder hidden state.
        encoder_outputs, encoder_final = self.encode(x, seq_len_x)
        hidden = self.init_decoder(encoder_outputs, encoder_final)

        # Compute the logits for P(y|x).
        outputs = []
        all_att_weights = []
        max_time = y.size(1)
        for t in range(max_time):
            prev_y = y[:, t]
            y_embed = self.tgt_embed(prev_y)
            pre_output, hidden, att_weights = self.decoder.step(y_embed, hidden, seq_mask_x,
                                                                encoder_outputs)
            logits = self.generate(pre_output)
            outputs.append(logits)
            all_att_weights.append(att_weights)

        return torch.cat(outputs, dim=1), torch.cat(all_att_weights, dim=1)

    def loss(self, logits, targets, reduction, label_smoothing=0.):
        """
        Computes the negative categorical log-likelihood for the given model output.

        :param logits: outputs of the model, the unnormalized probabilities [B, T, vocab_size]
        :param targets: target labels [B, T]
        :param reduction: what reduction to apply, none ([B]), mean ([]) or sum ([])
        """

        # Compute the loss for each batch element. Logits are of the form [B, T, vocab_size],
        # whereas the cross-entropy function wants a loss of the form [B, vocab_size, T].
        log_likelihood = - F.cross_entropy(logits.permute(0, 2, 1), targets, ignore_index=self.pad_idx, reduction="none")
        if label_smoothing > 0:
            # TODO Switch condnmt to the same loss as aevnmt (with torch.distributions).
            likelihood = Categorical(logits=logits)
            smooth_loss = label_smoothing_loss(likelihood, targets,
                                               ignore_index=self.pad_idx)
            log_likelihood = (1-label_smoothing) * log_likelihood + label_smoothing * smooth_loss
        loss = - log_likelihood.sum(dim=-1)

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        out_dict = {"loss": loss}
        return out_dict
