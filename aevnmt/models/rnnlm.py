import torch
import torch.nn as nn
import torch.nn.functional as F

from aevnmt.components import rnn_creation_fn

class RNNLM(nn.Module):

    def __init__(self, vocab_size, emb_size, hidden_size, pad_idx,
                 dropout, num_layers, cell_type, tied_embeddings, feed_z_size=0,embedder=None):
        """
        An RNN language model.
        """
        super().__init__()
        self.pad_idx = pad_idx
        self.hidden_size = hidden_size
        self.embedder = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx) if embedder is None else embedder
        rnn_dropout = 0. if num_layers == 1 else dropout
        rnn_fn = rnn_creation_fn(cell_type)
        self.rnn = rnn_fn(emb_size+feed_z_size, hidden_size, batch_first=True,
                          dropout=rnn_dropout, num_layers=num_layers)
        self.tied_embeddings = tied_embeddings
        if not tied_embeddings:
            self.output_matrix = nn.Parameter(torch.randn(vocab_size, hidden_size))
        self.dropout_layer = nn.Dropout(p=dropout)

    def step(self, x_embed, hidden,z=None):
        rnn_input = x_embed.unsqueeze(1)
        if z is not None:
            #z: (B, latent_size)
            #Concatenate z to RNN input at each timestep
            rnn_input=torch.cat([ rnn_input, z.unsqueeze(1)  ],dim=-1)
        rnn_output, hidden = self.rnn(rnn_input, hidden)
        rnn_output = self.dropout_layer(rnn_output)
        W_out = self.embedder.weight if self.tied_embeddings else self.output_matrix
        logits = F.linear(rnn_output, W_out)

        return hidden, logits

    def forward(self, x, hidden=None, z=None):
        x_embed = self.dropout_layer(self.embedder(x)) #(B, T, embed_size)

        outputs = []
        for t in range(x_embed.size(1)):
            rnn_input = x_embed[:, t].unsqueeze(1) #(B, 1, embed_size)
            if z is not None:
                #z: (B, latent_size)
                #Concatenate z to RNN input at each timestep
                rnn_input=torch.cat([ rnn_input, z.unsqueeze(1)  ],dim=-1)
            rnn_output, hidden = self.rnn(rnn_input, hidden)
            rnn_output = self.dropout_layer(rnn_output)
            W_out = self.embedder.weight if self.tied_embeddings else self.output_matrix
            logits = F.linear(rnn_output, W_out)
            outputs.append(logits)
        return torch.cat(outputs, dim=1)

    def loss(self, logits, targets, reduction="mean"):
        """
        Computes the negative categorical log-likelihood for the given model output.

        :param logits: outputs of the model, the unnormalized probabilities [B, T, vocab_size]
        :param targets: target labels [B, T]
        :param reduction: what reduction to apply, none ([B]), mean ([]) or sum ([])
        """

        # Compute the loss for each batch element. Logits are of the form [B, T, vocab_size],
        # whereas the cross-entropy function wants a loss of the form [B, vocab_svocab_sizee, T].
        logits = logits.permute(0, 2, 1)
        loss = F.cross_entropy(logits, targets, ignore_index=self.pad_idx, reduction="none")
        loss = loss.sum(dim=1)

        # Return differently according to the reduction setting.
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "none":
            return loss
        else:
            raise Exception(f"Unknown reduction option {reduction}")

    def NLL(self, logits, targets, reduction="mean"):
        return self.loss(logits, targets, reduction=reduction)
