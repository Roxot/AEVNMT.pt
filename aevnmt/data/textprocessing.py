import sacremoses
from subword_nmt.apply_bpe import BPE
from mosestokenizer import MosesSentenceSplitter
import codecs
import re


class SentenceSplitter:

    def __init__(self, language_code):
        self.language_code = language_code

    def split(self, data: list):
        data = [line for line in data if line.strip()]
        if len(data):
            with MosesSentenceSplitter(self.language_code) as splitter:
                data = splitter(data)
        return data


class TextProcess:

    def __call__(self, line: str) -> str:
        return line


class Tokenizer(TextProcess):

    def __init__(self, lang):
        self.tokenizer = sacremoses.MosesTokenizer(lang)

    def __call__(self, line: str) -> str:
        return self.tokenizer.tokenize(line, return_str=True)


class Detokenizer(TextProcess):

    def __init__(self, lang):
        self.detokenizer = sacremoses.MosesDetokenizer(lang)

    def __call__(self, line: str) -> str:
        return self.detokenizer.detokenize(line.split(), return_str=True)


class Lowercaser(TextProcess):

    def __init__(self, lang):
        pass

    def __call__(self, line: str) -> str:
        return line.lower()


class Truecaser(TextProcess):

    def __init__(self, truecase_model):
        self.truecaser = sacremoses.MosesTruecaser(truecase_model)

    def __call__(self, line: str) -> str:
        return self.truecaser.truecase(line, return_str=True)


class Recaser(TextProcess):

    def __init__(self, lang):
        self.recaser = sacremoses.MosesDetruecaser()

    def __call__(self, line: str) -> str:
        return self.recaser.detruecase(line, return_str=True)


class WordSegmenter(TextProcess):

    def __init__(self, bpe_codes, separator="@@", encoding='utf-8'):
        self.separator = separator.strip()
        self.bpe = BPE(
            codecs.open(bpe_codes, encoding=encoding),
            separator=self.separator)

    def __call__(self, line: str) -> str:
        return self.bpe.process_line(line)


class WordDesegmenter(TextProcess):

    def __init__(self, separator="@@", encoding='utf-8'):
        self.separator = separator.strip()

    def __call__(self, line: str) -> str:
        return re.sub(f"({self.separator} )|({self.separator} ?$)|( {self.separator})|(^ ?{self.separator})", "", line)


class Pipeline:

    def __init__(self, pre=[], post=[]):
        self._pre = pre
        self._post = post

    def pre(self, line: str) -> str:
        for module in self._pre:
            line = module(line)
        return line

    def post(self, line: str) -> str:
        for module in self._post:
            line = module(line)
        return line
