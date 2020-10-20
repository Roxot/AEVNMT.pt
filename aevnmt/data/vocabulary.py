from .constants import UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN

class Vocabulary:

    def __init__(self):
        self.idx_to_word = {0: UNK_TOKEN, 1: PAD_TOKEN, 2: SOS_TOKEN, 3: EOS_TOKEN}
        self.word_to_idx = {UNK_TOKEN: 0, PAD_TOKEN: 1, SOS_TOKEN: 2, EOS_TOKEN: 3}
        self.word_freqs = {}

    def __getitem__(self, key):
        """
        Returns the word id for a given word, or the id for UNK_TOKEN if not in the vocabulary.
        """
        return self.word_to_idx[key] if key in self.word_to_idx else self.word_to_idx[UNK_TOKEN]

    def word(self, idx):
        """
        Returns the word for a given id in the vocabulary.
        """
        return self.idx_to_word[idx]

    def size(self):
        return len(self.word_to_idx)

    def save(self, filename):
        with open(filename, "w") as f:
            for idx in range(len(self.idx_to_word)):
                if idx <= 3:
                    continue
                f.write(f"{self.idx_to_word[idx]}\n")

    def print_statistics(self):
        print(f"Vocabulary size: {len(self.word_to_idx):,} word types")
        print()

        if len(self.word_freqs) > 0:

            # Print the most frequent words.
            print("Top 10 most frequent words:")
            for word, freq in sorted(self.word_freqs.items(), key=lambda kv: kv[1],
                                     reverse=True)[:10]:
                print(f"{' '*3}{word:30}frequency = {freq:,}")
            print()

            # Print the least frequent words.
            print("Top 10 least frequent words:")
            for word, freq in sorted(self.word_freqs.items(), key=lambda kv: kv[1])[:10]:
                print(f"{' '*3}{word:30}frequency = {freq:,}")
            print()

        # Print the longest words.
        print("Top 10 longest words:")
        for word in sorted(self.word_to_idx.keys(), key=lambda w: len(w), reverse=True)[:10]:
            if word in self.word_freqs:
                freq = self.word_freqs[word]
                print(f"{' '*3}{word:30}length = {len(word):<5}frequency = {freq:,}")
            else:
                print(f"{' '*3}{word:30}length = {len(word):<5}")
        print()

        # Print the shortest words.
        print("Top 10 shortest words:")
        for word in sorted(self.word_to_idx.keys(), key=lambda w: len(w))[:10]:
            if word in self.word_freqs:
                freq = self.word_freqs[word]
                print(f"{' '*3}{word:30}length = {len(word):<5}frequency = {freq:,}")
            else:
                print(f"{' '*3}{word:30}length = {len(word):<5}")

    @staticmethod
    def from_data(filenames, min_freq=0, max_size=None):
        """
        Creates a vocabulary from a list of data files.

        This assumes that the data files have been tokenized and pre-processed.
        The dictionary ids are sorted in word frequency in increasing order.

        :param filenames: A list of filenames containing the data.
        :param min_freq: The minimum frequency of word occurrences in order to be included.
        :param max_size: The maximum vocabulary size (excluding special tokens).
        """
        vocab = Vocabulary()

        # Count word frequencies.
        for filename in filenames:
            with open(filename) as f:
                for line in f:
                    line = line.rstrip()
                    for word in line.split():
                        if word not in vocab.word_freqs:
                            vocab.word_freqs[word] = 1
                        else:
                            vocab.word_freqs[word] += 1

        # Fill the vocabulary based on word frequency.
        for word, freq in sorted(vocab.word_freqs.items(), key=lambda kv: kv[1], reverse=True):

            if freq >= min_freq and not (max_size > 0 and \
                    len(vocab.word_to_idx) == max_size+4) and word not in vocab.word_to_idx:
                idx = len(vocab.word_to_idx)
                vocab.word_to_idx[word] = idx
                vocab.idx_to_word[idx] = word
            else:
                del vocab.word_freqs[word]

        return vocab

    @staticmethod
    def from_file(vocab_file, max_size=None):
        """
        Creates a vocabulary from a file containing one word per line.

        :param vocab_file: File containing a single word per file.
        :param max_size: Maximum vocabulary size. Will only load the first max_size words
                         from the vocabulary file.
        """
        vocab = Vocabulary()
        with open(vocab_file) as f:
            for line in f:
                if max_size > 0 and len(vocab.word_to_idx) == max_size+4:
                    break

                word = line.rstrip()
                if word not in vocab.word_to_idx:
                    idx = len(vocab.word_to_idx)
                    vocab.word_to_idx[word] = idx
                    vocab.idx_to_word[idx] = word

        return vocab
