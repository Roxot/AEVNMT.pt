"""
Adapted from: https://github.com/joeynmt/joeynmt/blob/master/joeynmt/search.py
"""
from packaging import version

import torch
import torch.nn.functional as F
import numpy as np

from aevnmt.components import LuongDecoder

# from onmt
def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.
    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
    """
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
        .transpose(0, 1) \
        .repeat(count, 1) \
        .transpose(0, 1) \
        .contiguous() \
        .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x

def beam_search(decoder, tgt_embed_fn, generator_fn, tgt_vocab_size, hidden, encoder_outputs,
                encoder_final, seq_mask_x, sos_idx, eos_idx, pad_idx, beam_width, alpha,
                max_len,z=None):
    """
    Beam search with size beam_width. Follows OpenNMT-py implementation.
    In each decoding step, find the k most likely partial hypotheses.

    :param decoder: an initialized decoder
    """
    n_best = 1
    decoder.eval()
    with torch.no_grad():

        luong_decoding = isinstance(decoder, LuongDecoder)

        # Initialize the hidden state and create the initial input.
        batch_size = seq_mask_x.size(0)
        prev_y = torch.full(size=[batch_size], fill_value=sos_idx, dtype=torch.long,
                            device=seq_mask_x.device)
        if luong_decoding: hidden, prev_pre_output = hidden

        # Tile hidden decoder states and encoder outputs beam_width times
        hidden = tile(hidden, beam_width, dim=1)    # [layers, B*beam_width, H_dec]
        decoder.attention.proj_keys = tile(decoder.attention.proj_keys,
                                           beam_width, dim=0)
        encoder_outputs = tile(encoder_outputs.contiguous(), beam_width,
                               dim=0)               # [B*beam_width, T_x, H_enc]
        if z is not None:
            z=tile(z.contiguous(), beam_width,
                               dim=0)
        seq_mask_x = tile(seq_mask_x, beam_width, dim=0)    # [B*beam_width, 1, T_x]

        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=encoder_outputs.device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_width,
            step=beam_width,
            dtype=torch.long,
            device=encoder_outputs.device)
        alive_seq = torch.full(
            [batch_size * beam_width, 1],
            sos_idx,
            dtype=torch.long,
            device=encoder_outputs.device)

        # Give full probability to the first beam on the first step.
        topk_log_probs = (torch.tensor([0.0] + [float("-inf")] * (beam_width - 1),
                                       device=encoder_outputs.device).repeat(
                                        batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]
        results["scores"] = [[] for _ in range(batch_size)]
        results["gold_score"] = [0] * batch_size

        for step in range(max_len):
            prev_y = alive_seq[:, -1].view(-1)

            # expand current hypotheses, decode one single step
            if luong_decoding: hidden = (hidden, prev_pre_output)
            prev_y = tgt_embed_fn(prev_y)
            pre_output, hidden, _ = decoder.step(prev_y, hidden, seq_mask_x, encoder_outputs,z)
            logits = generator_fn(pre_output)
            if luong_decoding: hidden, prev_pre_output = hidden
            log_probs = F.log_softmax(logits, dim=-1).squeeze(1)  # [B*beam_width, |V_y|]

            # multiply probs by the beam probability (=add logprobs)
            log_probs += topk_log_probs.view(-1).unsqueeze(1)
            curr_scores = log_probs

            # compute length penalty
            if alpha > -1:
                length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
                curr_scores /= length_penalty

            # flatten log_probs into a list of possibilities
            curr_scores = curr_scores.reshape(-1, beam_width * tgt_vocab_size)

            # pick currently best top beam_width hypotheses (flattened order)
            topk_scores, topk_ids = curr_scores.topk(beam_width, dim=-1)

            if alpha > -1:
                # recover original log probs
                topk_log_probs = topk_scores * length_penalty

            # reconstruct beam origin and true word ids from flattened order
            if version.parse(torch.__version__) >= version.parse('1.3.0'):
                topk_beam_index = topk_ids.floor_divide(tgt_vocab_size)
            else:
                topk_beam_index = topk_ids.div(tgt_vocab_size)
            topk_ids = topk_ids.fmod(tgt_vocab_size)

            # map beam_index to batch_index in the flat representation
            batch_index = (
                topk_beam_index
                + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            # append latest prediction
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)  # batch_size*k x hyp_len

            is_finished = topk_ids.eq(eos_idx)
            if step + 1 == max_len:
                is_finished.fill_(1)

            # end condition is whether the top beam is finished
            end_condition = is_finished[:, 0].eq(1)

            # save finished hypotheses
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_width, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero(as_tuple=False).view(-1)
                    # store finished hypotheses for this batch
                    for j in finished_hyp:
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:])  # ignore start_token
                        )
                    # if the batch reached the end, save the n_best hypotheses
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        for n, (score, pred) in enumerate(best_hyp):
                            if n >= n_best:
                                break
                            results["scores"][b].append(score)
                            results["predictions"][b].append(pred)
                non_finished = end_condition.eq(0).nonzero(as_tuple=False).view(-1)

                # if all sentences are translated, no need to go further
                if len(non_finished) == 0:
                    break

                # remove finished batches for the next step
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))

            # reorder indices, outputs and masks
            select_indices = batch_index.view(-1)
            encoder_outputs = encoder_outputs.index_select(0, select_indices)
            if z is not None:
                z=z.index_select(0,select_indices)
            seq_mask_x = seq_mask_x.index_select(0, select_indices)
            decoder.attention.proj_keys = decoder.attention.proj_keys. \
                    index_select(0, select_indices)
            if luong_decoding:
                prev_pre_output = prev_pre_output.index_select(0, select_indices)

            if isinstance(hidden, tuple):
                # for LSTMs, states are tuples of tensors
                h, c = hidden
                h = h.index_select(1, select_indices)
                c = c.index_select(1, select_indices)
                hidden = (h, c)
            else:
                # for GRUs, states are single tensors
                hidden = hidden.index_select(1, select_indices)

    def pad_and_stack_hyps(hyps, pad_value):
        filled = np.ones((len(hyps), max([h.shape[0] for h in hyps])),
                         dtype=int) * pad_value
        for j, h in enumerate(hyps):
            for k, i in enumerate(h):
                filled[j, k] = i
        return filled

    # from results to stacked outputs
    # only works for n_best=1 for now
    assert n_best == 1

    final_outputs = pad_and_stack_hyps([r[0].cpu().numpy() for r in
                                        results["predictions"]],
                                        pad_value=pad_idx)

    return torch.from_numpy(final_outputs)


def beam_search_transformer(
        decoder, tgt_embed_fn, generator_fn, tgt_vocab_size, 
        encoder_outputs, seq_len_x, sos_idx, eos_idx, pad_idx,
        beam_width, alpha, max_len, z=None):
    """
    Beamsearch for the TransformerTM (TransformerDecoder). Follows OpenNMT-py implementation.

    :param decoder: An initialized TransformerDecoder.
    :param tgt_embed_fn: An embedder function, see TransformerTM.prepare_decoder_input.
    :param generator_fn: Function that generates logits from the TransformerDecoder output.
    :param tgt_vocab_size: Target language vocab size.
    :param encoder_outputs: Output state of the Encoder
    :param seq_len_x: Sequence lengths of encoder_outputs.
    :param alpha: Length penalty alpha
    :param z: Optional latent z, defaults to None
    """

    n_best = 1
    decoder.eval()
    with torch.no_grad():

        batch_size = seq_len_x.size(0)

        # Tile to [B*beam_width, ...]
        encoder_outputs = tile(encoder_outputs.contiguous(),
                               beam_width, dim=0)
        if z is not None:
            z = tile(z.contiguous(), beam_width, dim=0)
        seq_len_x = tile(seq_len_x, beam_width, dim=0)
        batch_offset = torch.arange(
            batch_size, 
            dtype=torch.long, 
            device=encoder_outputs.device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_width,
            step=beam_width,
            dtype=torch.long,
            device=encoder_outputs.device)
        alive_seq = torch.full(
            [batch_size * beam_width, 1],
            sos_idx,
            dtype=torch.long,
            device=encoder_outputs.device)

        # Give full probability to the first beam on the first step.
        topk_log_probs = (torch.tensor([0.0] + [float("-inf")] * (beam_width - 1),
                                       device=encoder_outputs.device).repeat(
                                       batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]
        results["scores"] = [[] for _ in range(batch_size)]
        results["gold_score"] = [0] * batch_size

        for step in range(max_len):
            prev_y = alive_seq

            # expand current hypotheses, decode one single step
            y_step, enc_out_step, seq_len_x_step = tgt_embed_fn(
                prev_y, encoder_outputs, seq_len_x, z
            )

            pre_output = decoder(y_step, enc_out_step, seq_len_x_step)
            logits = generator_fn(pre_output)
            logits = logits[:, -1] # Only keep logits of current step.
            log_probs = F.log_softmax(logits, dim=-1)

            log_probs += topk_log_probs.view(-1, 1)
            curr_scores = log_probs

            # compute length penalty
            if alpha > -1:
                length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
                curr_scores /= length_penalty

            # flatten log_probs into a list of possibilities
            curr_scores = curr_scores.reshape(-1, beam_width * tgt_vocab_size)

            # pick currently best top beam_width hypotheses (flattened order)
            topk_scores, topk_ids = curr_scores.topk(beam_width, dim=-1)

            if alpha > -1:
                # recover original log probs
                topk_log_probs = topk_scores * length_penalty

            # reconstruct beam origin and true word ids from flattened order
            if version.parse(torch.__version__) >= version.parse('1.3.0'):
                topk_beam_index = topk_ids.floor_divide(tgt_vocab_size)
            else:
                topk_beam_index = topk_ids.div(tgt_vocab_size)
            topk_ids = topk_ids.fmod(tgt_vocab_size)

            # map beam_index to batch_index in the flat representation
            batch_index = (
                topk_beam_index
                + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            # append latest prediction
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)  # batch_size*k x hyp_len

            is_finished = topk_ids.eq(eos_idx)
            if step + 1 == max_len:
                is_finished.fill_(1)

            # end condition is whether the top beam is finished
            end_condition = is_finished[:, 0].eq(1)

            # save finished hypotheses
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_width, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero(as_tuple=False).view(-1)
                    # store finished hypotheses for this batch
                    for j in finished_hyp:
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:])  # ignore start_token
                        )
                    # if the batch reached the end, save the n_best hypotheses
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        for n, (score, pred) in enumerate(best_hyp):
                            if n >= n_best:
                                break
                            results["scores"][b].append(score)
                            results["predictions"][b].append(pred)
                non_finished = end_condition.eq(0).nonzero(as_tuple=False).view(-1)

                # if all sentences are translated, no need to go further
                if len(non_finished) == 0:
                    break

                # remove finished batches for the next step
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))

            # reorder indices, outputs and masks
            select_indices = batch_index.view(-1)
            encoder_outputs = encoder_outputs.index_select(0, select_indices)
            if z is not None:
                z=z.index_select(0,select_indices)
            seq_len_x = seq_len_x.index_select(0, select_indices)

    def pad_and_stack_hyps(hyps, pad_value):
        filled = np.ones((len(hyps), max([h.shape[0] for h in hyps])),
                         dtype=int) * pad_value
        for j, h in enumerate(hyps):
            for k, i in enumerate(h):
                filled[j, k] = i
        return filled

    # from results to stacked outputs
    # only works for n_best=1 for now
    assert n_best == 1

    final_outputs = pad_and_stack_hyps([r[0].cpu().numpy() for r in
                                        results["predictions"]],
                                        pad_value=pad_idx)

    return torch.from_numpy(final_outputs)
            
