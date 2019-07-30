from torch.utils.data import Dataset
import numpy as np
from itertools import tee
from aevnmt.data.vocabulary import Vocabulary
from subword_nmt.apply_bpe import BPE
import codecs
import sys
import sacremoses
from aevnmt.data.textprocessing import TextProcess

class ParallelDataset(Dataset):

    def __init__(self, src_file, tgt_file, max_length=-1):
        self.data = []
        with open(src_file) as sf, open(tgt_file) as tf:
            for src, tgt in zip(sf, tf):
                src = src.strip()
                src_length = len(src.split())
                tgt = tgt.strip()
                tgt_length = len(tgt.split())
                if max_length < 0 or (src_length <= max_length and tgt_length <= max_length):
                    self.data.append((src, tgt))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ParallelDatasetFlippedView:

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        src, tgt = self.dataset[idx]
        return tgt, src


def split_list(x, size):
    output = []
    if size < 0:
        return [x]
    elif size == 0:
        raise ValueError("Use size -1 for no splitting or size more than 0.")
    while True:
        if len(x) > size:
            output.append(x[:size])
            x = x[size:]
        else:
            output.append(x)
            break
    return output


class TextDataset(Dataset):

    def __init__(self, filename, max_length=-1):
        self.data = []
        with open(filename) as f:
            for line in f:
                line = line.strip()
                tokens = line.split()
                if max_length < 0 or len(tokens) <= max_length:
                    self.data.append(line)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class InputTextDataset(Dataset):

    def __init__(self, generator, max_length=-1, split=False):
        self.data = []
        self.parts = []
        for line in generator:
            line = line.strip()  # strip \n
            tokens = line.split()
            if max_length < 0 or len(tokens) <= max_length:
                self.parts.append([len(self.data)])
                self.data.append(line)
            elif split:  # sentence splitting (saves metadata to restore line-alignment with input)
                parts = []
                for snt in split_list(tokens, max_length):
                    parts.append(len(self.data))
                    self.data.append(' '.join(snt))
                self.parts.append(parts)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def parts(self, idx):
        return self.parts[idx]

    def join(self, outputs):
        result = []
        for parts in self.parts:
            result.append(' '.join(outputs[idx] for idx in parts))
        return result

class RawInputTextDataset(Dataset):

    def __init__(self, filename, max_length=-1, split=False, tokenizer=None, bpe_codes=None, truecaser=None):
        self.tokenizer = sacremoses.MosesTokenizer(tokenizer) if tokenizer else None
        self.bpe = BPE(codecs.open(bpe_codes, encoding='utf-8')) if bpe_codes else None
        self.truecase = sacremoses.MosesTruecaser(truecaser) if truecaser else None
        self.data = []
        self.parts = []
        f = sys.stdin if filename == '-' else open(filename)
        for line in f:
            line = line.strip()  # strip \n
            if tokenizer:  # tokenize (returns string)
                #line = sacrebleu.tokenize_v14_international(line)
                line = self.tokenizer.tokenize(line, return_str=True)
            if truecaser:  # truecase using Moses truecaser
                line = self.truecase.truecase(line, return_str=True)
            if bpe_codes:  # segment (returns string)
                line = self.bpe.process_line(line)
            tokens = line.split()
            if max_length < 0 or len(tokens) <= max_length:
                self.parts.append([len(self.data)])
                self.data.append(line)
            elif split:  # sentence splitting (saves metadata to restore line-alignment with input)
                parts = []
                for snt in split_list(tokens, max_length):
                    parts.append(len(self.data))
                    self.data.append(' '.join(snt))
                self.parts.append(parts)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def parts(self, idx):
        return self.parts[idx]

    def join(self, outputs):
        result = []
        for parts in self.parts:
            result.append(' '.join(outputs[idx] for idx in parts))
        return result

def postprocess(outputs, truecaser=None, tokenizer=None):
    if truecaser: 
        tr = sacremoses.MosesDetruecaser()
        output = [tr.detruecase(hyp, return_str=True) for hyp in outputs]
    if tokenizer:
        tk = sacremoses.MosesDetokenizer(tokenizer)
        outputs = [tk.detokenize(hyp.split(), return_str=True) for hyp in outputs]
    return outputs
        

def basic_tokenize_parallel(files: list, max_length=-1):
    """
    Reads in parallel tuples and returns a generator of tuples of sentences (each a list of string tokens).
    Tokenization is done by string.split()

    :param files: each file in this list is one stream of the corpus.
    :param max_length: if more than 0, discards a tuple if any of its sentences is longer than the value specified
    :return: a generator of tuples of sentences (lists of string tokens)
    """

    handlers = []
    for path in files:
        handlers.append(open(path))

    for lines in zip(*handlers):
        sentences = [line.strip().split() for line in lines]
        if 0 < max_length < max(len(sentence) for sentence in sentences):
            continue
        yield sentences

    for h in handlers:
        h.close()


def iterate_view(generator, stream):
    """Returns one element of the tuple"""
    for t in generator:
        yield t[stream]


def construct_memmap(generator, lengths, output_path, dtype):
    """Stores a np.array with shape [nb_tokens] in disk"""
    nb_tokens = np.sum(lengths)

    # construct memory mapped array
    mmap = np.memmap(output_path, dtype=dtype, mode='w+', shape=nb_tokens)

    # prepare for populating memmap
    offset = 0
    offsets = []

    # populate memory map
    for seq, seq_len in zip(generator, lengths):
        offsets.append(offset)
        # here we have a valid sequence, thus we memory map it
        mmap[offset:offset + seq_len] = seq
        offset += seq_len

    del mmap
    return np.array(offsets, dtype=dtype)


def map_tokens_to_ids(generator, vocab: Vocabulary):
    """
    Wraps a generator around a mapping function from string tokens to token ids.
    :param generator:
    :param vocab:
    :return: generator
    """
    for sentence in generator:
        yield [vocab[word] for word in sentence]


class MemMappedCorpus(Dataset):

    def __init__(self, filename, memmap_path: str, vocab: Vocabulary, max_length=-1, dtype='int64', generator=None):
        """

        :param filename:
        :param memmap_path:
        :param vocab:
        :param max_length:
        :param dtype:
        :param generator: use it to bypass loading from file
        """
        if generator is None:
            generator = iterate_view(basic_tokenize_parallel([filename], max_length=max_length), stream=0)
        self.generator, iterator1, iterator2 = tee(generator, 3)
        self.lengths = np.array([len(s) for s in iterator1], dtype=dtype)
        self.memmap_path = memmap_path
        self.offsets = construct_memmap(map_tokens_to_ids(iterator2, vocab), self.lengths, memmap_path, dtype=dtype)
        self.mmap = np.memmap(self.memmap_path, dtype=dtype, mode='r')
        self.vocab = vocab

    def __len__(self):
        return len(self.lengths)

    def as_memmap(self, idx):
        offset = self.offsets[idx]
        return self.mmap[offset: offset + self.lengths[idx]].tolist()

    def as_string(self, idx):
        return ' '.join(self.vocab.word(t) for t in self.as_memmap(idx))

    def __getitem__(self, idx):
        return self.as_string(idx)


class MemMappedParallelCorpus(Dataset):

    def __init__(self, filenames: list, memmap_path: str, vocabs: 'list[Vocabulary]', max_length=-1, dtype='int64'):
        self.nb_streams = len(filenames)
        generator = basic_tokenize_parallel(filenames, max_length=max_length)
        iterators = tee(generator, self.nb_streams + 1)
        self.generator = iterators[0]

        self.corpora = []
        for i, it in enumerate(iterators[1:]):
            corpus = MemMappedCorpus(
                None, f"{memmap_path}{i}", vocabs[i],
                max_length=-1,
                dtype=dtype,
                generator=iterate_view(it, i))
            self.corpora.append(corpus)

    def __len__(self):
        return len(self.corpora[0])

    def as_memmap(self, idx):
        return [corpus.as_memmap(idx) for corpus in self.corpora]

    def as_string(self, idx):
        return [corpus.as_string(idx) for corpus in self.corpora]

    def __getitem__(self, idx):
        return [corpus[idx] for corpus in self.corpora]
