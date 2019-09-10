import torch
from torch.distributions.categorical import Categorical


def ancestral_sample(decoder, tgt_embed_fn, generator_fn, hidden, encoder_outputs,
                  encoder_final, seq_mask_x, sos_idx, eos_idx, pad_idx, max_len, greedy=False):
    """
    :param decoder: an instance of aevnmt.components.LuongDecoder or
                    aevnmt.components.BahdanauDecoder that has been
                    initialized.
    :param tgt_embed_fn: a function to embed target words
    :param generator_fn: a function that generates the logits from the rnn outputs
    :param hidden: the initial decoder hidden state
    :param encoder_outputs: the encoder outputs
    :param encoder_final: the final encoder state
    :param seq_mask_x: the source sentence position mask
    :param sos_idx: the start-of-index id
    :param pad_idx: the pad token id
    :param max_len: the maximum sentence length to decode
    """
    # Initialize the hidden state and create the initial input.
    batch_size = seq_mask_x.size(0)
    prev_y = torch.full(size=[batch_size], fill_value=sos_idx, dtype=torch.long,
                        device=seq_mask_x.device)
 
    # Decode step-by-step by picking the maximum probability word
    # at each time step.
    predictions = []
    log_probs = []
    is_complete = torch.zeros_like(prev_y).unsqueeze(-1).byte()
    for t in range(max_len):
        prev_y = tgt_embed_fn(prev_y)
        pre_output, hidden, _ = decoder.step(prev_y, hidden, seq_mask_x, encoder_outputs)
        logits = generator_fn(pre_output)
        py_x = Categorical(logits=logits)
        if greedy:
            prediction = torch.argmax(logits, dim=-1)
        else:
            prediction = py_x.sample()
        prev_y = prediction.view(batch_size)
        log_prob_pred = py_x.log_prob(prediction)
        log_probs.append(torch.where(is_complete, torch.zeros_like(log_prob_pred), log_prob_pred))
        predictions.append(torch.where(is_complete, torch.full_like(prediction, pad_idx), prediction))
        is_complete = is_complete | (prediction == eos_idx).byte()

    return {'sample': torch.cat(predictions, dim=1), 'log_probs': torch.cat(log_probs, dim=1)}


def greedy_decode(decoder, tgt_embed_fn, generator_fn, hidden, encoder_outputs,
                  encoder_final, seq_mask_x, sos_idx, eos_idx, pad_idx, max_len):
    decoder.eval()
    with torch.no_grad():
        d = ancestral_sample(decoder, tgt_embed_fn, generator_fn, hidden, encoder_outputs,
                      encoder_final, seq_mask_x, sos_idx, eos_idx, pad_idx, max_len, greedy=True)
    # TODO: prepare other functions to expect a dictionary
    return d['sample']

def sampling_decode(decoder, tgt_embed_fn, generator_fn, hidden, encoder_outputs,
                  encoder_final, seq_mask_x, sos_idx, eos_idx, pad_idx, max_len):
    decoder.eval()
    with torch.no_grad():
        d = ancestral_sample(decoder, tgt_embed_fn, generator_fn, hidden, encoder_outputs,
                      encoder_final, seq_mask_x, sos_idx, eos_idx, pad_idx, max_len, greedy=False)
    return d['sample']
