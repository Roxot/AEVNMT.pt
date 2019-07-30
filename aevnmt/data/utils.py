import re
import torch
import numpy as np
import sys

from .constants import UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN


def create_noisy_batch(sentences, vocab, device, word_dropout=0., map_to_ids=True):
    """
    Converts a list of sentences to a padded batch of word ids. Returns
    an input batch, an output batch shifted by one, a sequence mask over
    the input batch, and a tensor containing the sequence length of each
    batch element.
    :param sentences: a list of sentences, each a list of token ids
    :param vocab: a Vocabulary object for this dataset
    :param device:
    :param word_dropout: rate at which we omit words from the context (input)
    :returns: a batch of padded inputs, a batch of padded outputs, mask, lengths
    """

    # sentences is a list of np arrays with int64 in it
    if map_to_ids:
        sentences = [[vocab[w] for w in sen.split()] for sen in sentences]
    # from here sentences are already made of token ids
    tok = np.array([[vocab[SOS_TOKEN]] + sen + [vocab[EOS_TOKEN]] for sen in sentences])
    seq_lengths = [len(sen)-1 for sen in tok]
    max_len = max(seq_lengths)
    pad_id = vocab[PAD_TOKEN]
    pad_id_input = [
        [sen[t] if t < seq_lengths[idx] else pad_id for t in range(max_len)]
            for idx, sen in enumerate(tok)]

    # Replace words of the input with <unk> with p = word_dropout.
    if word_dropout > 0.:
        unk_id = vocab[UNK_TOKEN]
        noisy_input = [
            [unk_id if (np.random.random() < word_dropout and t < seq_lengths[idx]) else word_ids[t] for t in range(max_len)]
                for idx, word_ids in enumerate(pad_id_input)]
    else:
        noisy_input = pad_id_input

    # The output batch is shifted by 1.
    pad_id_output = [
        [sen[t+1] if t < seq_lengths[idx] else pad_id for t in range(max_len)]
            for idx, sen in enumerate(tok)]

    # Convert everything to PyTorch tensors.
    batch_input = torch.tensor(pad_id_input)
    batch_noisy_input = torch.tensor(noisy_input)
    batch_output = torch.tensor(pad_id_output)
    seq_mask = (batch_input != vocab[PAD_TOKEN])
    seq_length = torch.tensor(seq_lengths)

    # Move all tensors to the given device.
    batch_input = batch_input.to(device)
    batch_noisy_input = batch_noisy_input.to(device)
    batch_output = batch_output.to(device)
    seq_mask = seq_mask.to(device)
    seq_length = seq_length.to(device)

    return batch_input, batch_output, seq_mask, seq_length, batch_noisy_input


def create_batch(sentences, vocab, device, map_to_ids=True):
    batch_input, batch_output, seq_mask, seq_length, _ = create_noisy_batch(
        sentences, vocab, device, word_dropout=0., map_to_ids=map_to_ids)
    return batch_input, batch_output, seq_mask, seq_length


def batch_to_sentences(tensors, vocab, no_filter=False):
    """
    Converts a batch of word ids back to sentences.
    :param tensors: [B, T] word ids
    :param vocab: a Vocabulary object for this dataset
    :param no_filter: whether to filter sos, eos, and pad tokens.
    :returns: an array of strings (each a sentence)
    """
    sentences = []
    batch_size = tensors.size(0)
    for idx in range(batch_size):
        sentence = [vocab.word(t.item()) for t in tensors[idx,:]]

        # Filter out the start-of-sentence and padding tokens.
        if not no_filter:
            sentence = list(filter(lambda t: t != PAD_TOKEN and t != SOS_TOKEN, sentence))

        # Remove the end-of-sentence token and all tokens following it.
        if EOS_TOKEN in sentence and not no_filter:
            sentence = sentence[:sentence.index(EOS_TOKEN)]

        sentences.append(" ".join(sentence))
    return np.array(sentences)

def remove_subword_tokens(sentences, subword_token):
    """
    Removes all subword tokens from a list of sentences. E.g. "The bro@@ wn fox ." with
    subword_token="@@" will be turned into "The brown fox .".

    :param sentences: a list of sentences
    :param subword_token: the subword token.
    """
    subword_token = subword_token.strip()
    clean_sentences = []
    for sentence in sentences:
        clean_sentences.append(re.sub(f"({subword_token} )|({subword_token} ?$)|"
                               f"( {subword_token})|(^ ?{subword_token})", "",
                               sentence))
    return clean_sentences
