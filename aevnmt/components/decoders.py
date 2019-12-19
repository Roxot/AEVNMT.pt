import torch
import torch.nn as nn

from .utils import rnn_creation_fn, tile_rnn_hidden

def tile_rnn_hidden_for_decoder(hidden, decoder):
    rnn = decoder.rnn
    hidden_state = tile_rnn_hidden(hidden, rnn)
    if isinstance(decoder, LuongDecoder):
        return (hidden_state, None)
    else:
        return hidden_state

class BahdanauDecoder(nn.Module):

    def __init__(self, emb_size, hidden_size, attention,
            dropout=0., num_layers=1, cell_type="lstm", init_from_encoder_final=True, 
            feed_z_size=0):
        """
        RNN decoder with Bahdanau style updates.
        """
        super().__init__()
        self.init_from_encoder_final = init_from_encoder_final
        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.feed_z_size = feed_z_size
        rnn_dropout = 0. if num_layers == 1 else dropout
        rnn_fn = rnn_creation_fn(cell_type)
        self.rnn = rnn_fn(emb_size + feed_z_size + attention.key_size, hidden_size, batch_first=True,
                          dropout=rnn_dropout, num_layers=num_layers)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(hidden_size + feed_z_size + emb_size + attention.key_size,
                                          hidden_size, bias=True)
        self.attention = attention

        if init_from_encoder_final:
            self.init_layer = nn.Linear(attention.key_size, hidden_size)

    def init_decoder(self, encoder_outputs, encoder_final):

        # Pre-compute the attention keys.
        self.attention.compute_proj_keys(encoder_outputs)

        if self.init_from_encoder_final:
            hidden = self.init_layer(encoder_final)
            hidden = hidden.unsqueeze(0) # [B, H_enc_total] -> [1, B,  H_enc_total]
            hidden = tile_rnn_hidden(hidden, self.rnn) # [num_layers, B, H_enc_total]
            return hidden

    def step(self, prev_embed, hidden, x_mask, encoder_outputs, z=None):
        """
        :param prev_embed: [B, emb_size]
        :param hidden: [num_layers, 1, hidden_size]
        :param x_mask:
        :param encoder_outputs:
        """

        # For the query we use h for LSTMs.
        if self.cell_type == "lstm":
            query = hidden[0]
        else:
            query = hidden

        if self.feed_z_size > 0:
            #z: (B, latent_size)
            #Concatenate z to RNN input at each timestep
            prev_embed=torch.cat([ prev_embed, z ],dim=-1)

        # Compute the context vector.
        query = query[-1].unsqueeze(1)
        prev_embed = prev_embed.unsqueeze(1)
        context, att_weights = self.attention(query, x_mask, encoder_outputs)
        
        # Update the RNN hidden state.
        rnn_input = torch.cat([prev_embed, context], dim=-1)
        rnn_output, hidden = self.rnn(rnn_input, hidden)

        # Compute the pre-outputs.
        pre_output = torch.cat([prev_embed, rnn_output, context], dim=-1)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        return pre_output, hidden, att_weights

    def forward(self, y_embed, x_mask, encoder_outputs, encoder_final, hidden=None, z=None):
        """
        Does teacher forcing. Unrolls entire RNN.
        """
        if self.init_from_encoder_final:
            hidden = self.init_decoder(encoder_outputs, encoder_final)
        else:
            self.init_decoder(encoder_outputs, encoder_final)

        outputs = []
        all_att_weights = []
        max_time = y_embed.size(1)
        for t in range(max_time):
            prev_embed = y_embed[:, t]
            pre_output, hidden, att_weights = self.step(prev_embed, hidden, x_mask, encoder_outputs, z=z)
            outputs.append(pre_output)
            all_att_weights.append(att_weights)
        return torch.cat(outputs, dim=1), torch.cat(all_att_weights, dim=1)


class LuongDecoder(nn.Module):

    def __init__(self, emb_size, hidden_size, attention,
                 dropout=0., num_layers=1, cell_type="lstm", init_from_encoder_final=True, feed_z_size=0):
        """
        RNN decoder with Luong style updates.
        """
        super().__init__()
        self.init_from_encoder_final = init_from_encoder_final
        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.feed_z_size = feed_z_size
        rnn_dropout = 0. if num_layers == 1 else dropout
        rnn_fn = rnn_creation_fn(cell_type)
        self.rnn = rnn_fn(emb_size + feed_z_size + hidden_size, hidden_size, batch_first=True,
                          dropout=rnn_dropout, num_layers=num_layers)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(hidden_size + attention.key_size,
                                          hidden_size, bias=True)
        self.attention = attention

        if init_from_encoder_final:
            self.init_layer = nn.Linear(attention.key_size, hidden_size)

    def init_decoder(self, encoder_outputs, encoder_final):
        """
        :param encoder_outputs: encoder outputs to initialize the attention mechanism.
        :param encoder_final: encoder final state

        Returns a tuple (hidden, None), which can be used as input_vectors to the step function.
        """

        # Pre-compute the attention keys.
        self.attention.compute_proj_keys(encoder_outputs)

        if self.init_from_encoder_final:
            hidden = self.init_layer(encoder_final)
            hidden = hidden.unsqueeze(0) # [B, H_enc_total] -> [1, B,  H_enc_total]
            hidden = tile_rnn_hidden(hidden, self.rnn) # [num_layers, B, H_enc_total]
            return (hidden, None)

    def step(self, prev_embed, input_vectors, x_mask, encoder_outputs, z=None):
        """
        :param prev_embed: [B, emb_size]
        :param hidden: [num_layers, 1, hidden_size]
        :param input_vectors: a tuple (hidden, prev_pre_output) containing the rnn hidden state
                              and the previous pre-output vector, at the first time step the
                              previous pre-output vector should be None.
                              ([num_layers, 1, hidden_size], [B, 1, hidden_size])
        :param x_mask:
        :param encoder_outputs:
        """

        # Obtain the hidden state and previous pre-output vector.
        hidden, prev_pre_output = input_vectors

        # Initialize the pre-output vector with zeros.
        if prev_pre_output is None:
            prev_pre_output = torch.zeros_like(prev_embed).unsqueeze(1)

        if self.feed_z_size > 0:
            #z: (B, latent_size)
            #Concatenate z to RNN input at each timestep
            prev_embed=torch.cat([ prev_embed, z ],dim=-1)

        # Update the RNN hidden state.
        prev_embed = prev_embed.unsqueeze(1)
        rnn_input = torch.cat([prev_embed, prev_pre_output], dim=-1)
        _, hidden = self.rnn(rnn_input, hidden)

        # For the query we use h for LSTMs.
        if self.cell_type == "lstm":
            query = hidden[0]
        else:
            query = hidden
        query = query[-1].unsqueeze(1)

        # Compute the context vector.
        context, att_weights = self.attention(query, x_mask, encoder_outputs)

        # Compute the pre-output vector.
        pre_output = torch.cat([query, context], dim=-1)
        pre_output = self.dropout_layer(pre_output)
        pre_output = torch.tanh(self.pre_output_layer(pre_output))

        return pre_output, (hidden, pre_output), att_weights

    def forward(self, y_embed, x_mask, encoder_outputs, encoder_final, hidden=None, z=None):
        """
        Does teacher forcing. Unrolls entire RNN.
        """
        if self.init_from_encoder_final:
            hidden = self.init_decoder(encoder_outputs, encoder_final)
        else:
            self.init_decoder(encoder_outputs, encoder_final)
            hidden = (hidden, None)

        outputs = []
        all_att_weights = []
        max_time = y_embed.size(1)
        for t in range(max_time):
            prev_embed = y_embed[:, t]
            pre_output, hidden, att_weights = self.step(prev_embed, hidden, x_mask, encoder_outputs, z=z)
            outputs.append(pre_output)
            all_att_weights.append(att_weights)
        return torch.cat(outputs, dim=1), torch.cat(all_att_weights, dim=1)

