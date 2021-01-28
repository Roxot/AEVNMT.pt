import torch
import numpy as np
import sys
import time

from aevnmt.hparams import Hyperparameters
from aevnmt.data import TextDataset, RawInputTextDataset, remove_subword_tokens, postprocess
from aevnmt.train import create_model
from aevnmt.train_utils import load_vocabularies, compute_bleu
from aevnmt.data.datasets import InputTextDataset
from aevnmt.data.textprocessing import SentenceSplitter
from aevnmt.data.textprocessing import Pipeline
from aevnmt.data.textprocessing import Tokenizer, Detokenizer
from aevnmt.data.textprocessing import Lowercaser, Truecaser, Recaser
from aevnmt.data.textprocessing import WordSegmenter, WordDesegmenter

from torch.utils.data import DataLoader
from pathlib import Path


class TranslationEngine:

    def __init__(self, hparams):

        output_dir = Path(hparams.output_dir)
        verbose = hparams.verbose

        if hparams.vocab.prefix is None:
            hparams.vocab.prefix = str(output_dir / "vocab")
            hparams.vocab.shared = False

        # Select the correct device (GPU or CPU).
        device = torch.device("cuda:0") if hparams.use_gpu else torch.device("cpu")

        # Pre/post-processing
        if hparams.tokenize:
            src_tokenizer_lang = hparams.src
            tgt_tokenizer_lang = hparams.tgt
        else:
            src_tokenizer_lang = tgt_tokenizer_lang = None
        if hparams.bpe.codes_prefix:
            src_bpe_codes = f"{hparams.bpe.codes_prefix}.{hparams.src}"
            tgt_bpe_codes = f"{hparams.bpe.codes_prefix}.{hparams.tgt}"
        else:
            src_bpe_codes = tgt_bpe_codes = None

        if hparams.lowercase and hparams.truecaser_prefix:
            raise ValueError("You cannot use lowercasing and truecasing at the same time")

        if hparams.truecaser_prefix:
            src_truecase_model = f"{hparams.truecaser_prefix}.{hparams.src}"
            tgt_truecase_model = f"{hparams.truecaser_prefix}.{hparams.tgt}"
        else:
            src_truecase_model = tgt_truecase_model = None

        model_checkpoint = output_dir / f"model/{hparams.criterion}/{hparams.src}-{hparams.tgt}.pt"

        self.hparams = hparams
        self.verbose = verbose
        self.device = device
        self.model_checkpoint = model_checkpoint
        self.src_tokenizer_lang = src_tokenizer_lang
        self.tgt_tokenizer_lang = tgt_tokenizer_lang
        self.src_bpe_codes = src_bpe_codes
        self.tgt_bpe_codes = tgt_bpe_codes
        self.src_truecase_model = src_truecase_model
        self.tgt_truecase_model = tgt_truecase_model
        self.pipeline = Pipeline()
        self.vocab_src = None
        self.vocab_tgt = None
        self.model = None
        self.translate_fn = None
        self.n_translated = 0

    @staticmethod
    def make_pipeline(hparams):
        # Loading pre/post-processing models
        if hparams.verbose:
            print("Loading pre/post-processing models", file=sys.stderr)

        preprocess = []
        postprocess = []

        # Tokenization
        if hparams.tokenize:
            preprocess.append(Tokenizer(hparams.src))
        if hparams.detokenize:
            postprocess.append(Detokenizer(hparams.tgt))

        # Case
        if hparams.lowercase and hparams.truecaser_prefix:
            raise ValueError("You cannot set --lowercase to true and provide a --truecaser_prefix at the same time")

        if hparams.lowercase:
            preprocess.append(Lowercaser(hparams.src))

        if hparams.truecaser_prefix:
            preprocess.append(Truecaser(f"{hparams.truecaser_prefix}.{hparams.src}"))
        if hparams.recase:
            postprocess.append(Recaser(hparams.tgt))

        # Word segmentation
        if hparams.bpe.codes_prefix:
            preprocess.append(WordSegmenter(f"{hparams.bpe.codes_prefix}.{hparams.src}", separator=hparams.subword_token))
        if hparams.bpe.merge:
            postprocess.append(WordDesegmenter(separator=hparams.subword_token))

        return Pipeline(pre=preprocess, post=list(reversed(postprocess)))

    def load_statics(self):
        # Loading vocabulary
        if self.verbose:
            t0 = time.time()
            print(f"Loading vocabularies src={self.hparams.src} tgt={self.hparams.tgt}", file=sys.stderr)
        self.vocab_src, self.vocab_tgt = load_vocabularies(self.hparams)

        # Load pre/post processing models and configure a pipeline
        self.pipeline = TranslationEngine.make_pipeline(self.hparams)

        if self.verbose:
            print(f"Restoring model selected wrt {self.hparams.criterion} from {self.model_checkpoint}", file=sys.stderr)

        model, _, _, translate_fn = create_model(self.hparams, self.vocab_src, self.vocab_tgt)
        
        if self.hparams.use_gpu:
            model.load_state_dict(torch.load(self.model_checkpoint))
        else:
            model.load_state_dict(torch.load(self.model_checkpoint, map_location='cpu'))

        self.model = model.to(self.device)
        self.translate_fn = translate_fn
        self.model.eval()
        if self.verbose:
            print("Done loading in %.2f seconds" % (time.time() - t0), file=sys.stderr)

    def translate(self, lines: list, stdout=sys.stdout):
        hparams = self.hparams
        if hparams.split_sentences:  # This is a type of pre-processing we do not a post-processing counterpart for
            if hparams.verbose:
                print(f"Running sentence splitter for {len(lines)} lines")
            lines = SentenceSplitter(hparams.src).split(lines)
            if hparams.verbose:
                print(f"Produced {len(lines)} sentences")
        if not lines:  # we do not like empty jobs
            return []
        input_data = InputTextDataset(
            generator=(self.pipeline.pre(line) for line in lines),
            max_length=hparams.max_sentence_length,
            split=True)
        input_dl = DataLoader(
            input_data, batch_size=hparams.batch_size,
            shuffle=False, num_workers=4)
        input_size = len(input_data)

        # Translate the data.
        num_translated = 0
        all_hypotheses = []
        if self.verbose:
            print(f"Translating {input_size} sentences...", file=sys.stderr)

        for input_sentences in input_dl:

            # Sort the input sentences from long to short.
            input_sentences = np.array(input_sentences)
            seq_len = np.array([len(s.split()) for s in input_sentences])
            sort_keys = np.argsort(-seq_len)
            input_sentences = input_sentences[sort_keys]

            t1 = time.time()
            # Translate the sentences using the trained model.
            hypotheses = self.translate_fn(
                self.model, input_sentences,
                self.vocab_src, self.vocab_tgt,
                self.device, hparams)

            num_translated += len(input_sentences)

            # Restore the original ordering.
            inverse_sort_keys = np.argsort(sort_keys)
            all_hypotheses += hypotheses[inverse_sort_keys].tolist()

            if self.verbose:
                print(f"{num_translated}/{input_size} sentences translated in {time.time() - t1:.2f} seconds.", file=sys.stderr)

        if hparams.show_raw_output:
            for i in range(len(input_data)):
                print(i + self.n_translated, '|||', input_data[i], '|||', all_hypotheses[i], file=sys.stderr)

        if hparams.max_sentence_length > 0:  # join sentences that might have been split
            all_hypotheses = input_data.join(all_hypotheses)

        # Post-processing
        all_hypotheses = [self.pipeline.post(h) for h in all_hypotheses]

        if stdout is not None:
            for hypothesis in all_hypotheses:
                print(hypothesis, file=stdout)

        self.n_translated += len(input_data)

        return all_hypotheses

    def interactive_translation_n(self, generator=sys.stdin, wait_for=1, stdout=sys.stdout):
        if self.verbose:
            print(f"Ready to start translating {wait_for} sentences at a time", file=sys.stderr)
        job = []
        for line in generator:
            job.append(line)
            if len(job) >= wait_for:
                self.translate(job, stdout=stdout)
                job = []
            if self.verbose:
                print(f"Waiting for {wait_for - len(job)} sentences", file=sys.stderr)

    def interactive_translation(self, generator=sys.stdin, stdout=sys.stdout):
        if self.verbose:
            print("Ready to start", file=sys.stderr)
        for i, line in enumerate(generator):
            self.translate([line], stdout=stdout)

    def translate_file(self, input_path, output_path=None, reference_path=None, stdout=None):
        if output_path is None:
            stdout = sys.stdout

        with open(input_path) as f:  # TODO: optionally segment input file into slices of n lines each
            translations = self.translate(f.readlines(), stdout=stdout)
            # If a reference set is given compute BLEU score.
            if reference_path is not None:
                ref_sentences = TextDataset(reference_path).data
                if self.hparams.postprocess_ref:
                    ref_sentences = [self.pipeline.post(r) for r in ref_sentences]
                bleu = compute_bleu(translations, ref_sentences, subword_token=None)
                print(f"\nBLEU = {bleu:.4f}")

            # If an output file is given write the output to that file.
            if output_path is not None:
                if self.verbose:
                    print(f"\nWriting translation output to {output_path}", file=sys.stderr)
                with open(output_path, "w") as f:
                    for translation in translations:
                        f.write(f"{translation}\n")

    def translate_stdin(self, stdout=sys.stdout):
        lines = [line for line in sys.stdin]
        self.translate(lines, stdout=stdout)


def main(hparams=None):
    # Load command line hyperparameters (and if provided from an hparams_file).
    if hparams is None:
        if "--help" not in sys.argv and "--hparams_file" not in sys.argv:
            # TODO This is added to prevent incorrect overriding of arguments, see Issue #14
            # When resolved, hparams.update_from_file can be used instead.
            output_dir = Path(sys.argv[sys.argv.index("--output_dir") + 1])
            hparams_file = str(output_dir / "hparams")
            sys.argv = [sys.argv[0]] + ['--hparams_file', hparams_file] + sys.argv[1:]
        hparams = Hyperparameters(check_required=False)

    engine = TranslationEngine(hparams)

    engine.load_statics()

    if hparams.translation.interactive > 0:
        if hparams.translation.interactive == 1:
            engine.interactive_translation()
        else:
            engine.interactive_translation_n(wait_for=hparams.translation.interactive)
    elif hparams.translation.input_file == '-':
        engine.translate_stdin()
    else:
        if hparams.translation.ref_file and hparams.split_sentences:
            raise ValueError("If you enable sentence splitting you will compromise line-alignment with the reference")
        engine.translate_file(
            input_path=hparams.translation.input_file,
            output_path=hparams.translation.output_file,
            reference_path=hparams.translation.ref_file
        )

if __name__ == "__main__":
    main()
