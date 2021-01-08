import torch
import numpy as np
import sys
import time

from aevnmt.hparams import Hyperparameters
from aevnmt.data import TextDataset, RawInputTextDataset, remove_subword_tokens, postprocess
from aevnmt.train_monolingual import create_model
from aevnmt.train_utils import load_vocabularies, compute_bleu
from aevnmt.data.datasets import InputTextDataset
from aevnmt.data.textprocessing import SentenceSplitter
from aevnmt.data.textprocessing import Pipeline
from aevnmt.data.textprocessing import Tokenizer, Detokenizer
from aevnmt.data.textprocessing import Lowercaser, Truecaser, Recaser
from aevnmt.data.textprocessing import WordSegmenter, WordDesegmenter

from torch.utils.data import DataLoader
from pathlib import Path


class GenerationEngine:

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
        else:
            src_tokenizer_lang = None

        if hparams.lowercase and hparams.truecaser_prefix:
            raise ValueError("You cannot use lowercasing and truecasing at the same time")

        model_checkpoint = output_dir / f"model/{hparams.criterion}/{hparams.src}.pt"

        self.hparams = hparams
        self.verbose = verbose
        self.device = device
        self.model_checkpoint = model_checkpoint
        self.src_tokenizer_lang = src_tokenizer_lang
        self.pipeline = Pipeline()
        self.vocab_src = None
        self.model = None
        self.translate_fn = None
        self.n_translated = 0

    @staticmethod
    def make_pipeline(hparams):
        # Loading pre/post-processing models
        if hparams.verbose:
            print("Loading pre/post-processing models", file=sys.stderr)

        postprocess = []

        # Tokenization
        if hparams.detokenize:
            postprocess.append(Detokenizer(hparams.tgt))

        # Case
        if hparams.lowercase and hparams.truecaser_prefix:
            raise ValueError("You cannot set --lowercase to true and provide a --truecaser_prefix at the same time")

        if hparams.recase:
            postprocess.append(Recaser(hparams.tgt))

        # Word segmentation
        if hparams.bpe.merge:
            postprocess.append(WordDesegmenter(separator=hparams.subword_token))

        return Pipeline(pre=[], post=list(reversed(postprocess)))

    def load_statics(self):
        # Loading vocabulary
        if self.verbose:
            t0 = time.time()
            print(f"Loading vocabularies src={self.hparams.src} tgt={self.hparams.tgt}", file=sys.stderr)
        self.vocab_src = load_vocabularies_monolingual(self.hparams)

        # Load pre/post processing models and configure a pipeline
        self.pipeline = GenerationEngine.make_pipeline(self.hparams)

        if self.verbose:
            print(f"Restoring model selected wrt {self.hparams.criterion} from {self.model_checkpoint}", file=sys.stderr)

        model, _, _, translate_fn = create_model(self.hparams, self.vocab_src)

        if self.hparams.use_gpu:
            model.load_state_dict(torch.load(self.model_checkpoint))
        else:
            model.load_state_dict(torch.load(self.model_checkpoint, map_location='cpu'))

        self.model = model.to(self.device)
        self.translate_fn = translate_fn
        self.model.eval()
        if self.verbose:
            print("Done loading in %.2f seconds" % (time.time() - t0), file=sys.stderr)

    def translate(self, num_samples: int, stdout=sys.stdout):
        hparams = self.hparams
        batch_size=hparams.batch_size

        # Translate the data.
        num_translated = 0
        all_hypotheses = []
        if self.verbose:
            print(f"Sampling {num_samples} sentences...", file=sys.stderr)

        num_batches=num_samples//batch_size
        if num_samples % batch_size > 0:
            num_batches+=1

        for batch_idx in range(num_batches):
            local_batch_size=batch_size
            if batch_idx == num_batches -1 and num_samples % batch_size > 0:
                local_batch_size=num_samples % batch_size

            t1 = time.time()
            # Translate the sentences using the trained model.
            hypotheses = self.translate_fn(
                self.model, local_batch_size,
                self.vocab_src,
                self.device, hparams)

            num_translated += local_batch_size

            # Restore the original ordering.
            all_hypotheses += hypotheses.tolist()

            if self.verbose:
                print(f"{num_translated}/{input_size} sentences translated in {time.time() - t1:.2f} seconds.", file=sys.stderr)

        if hparams.show_raw_output:
            for i in range(num_samples):
                print(i + self.n_translated, '|||' '|||', all_hypotheses[i], file=sys.stderr)

        # Post-processing
        all_hypotheses = [self.pipeline.post(h) for h in all_hypotheses]

        if stdout is not None:
            for hypothesis in all_hypotheses:
                print(hypothesis, file=stdout)

        self.n_translated += num_samples

        return all_hypotheses

    def generate_file(self, output_path=None, num_samples=100, stdout=None):
        if output_path is None:
            stdout = sys.stdout

        #TODO: add option to read latent code from file
        translations = self.generate(num_samples, stdout=stdout)

        # If an output file is given write the output to that file.
        if output_path is not None:
            if self.verbose:
                print(f"\nWriting translation output to {output_path}", file=sys.stderr)
            with open(output_path, "w") as f:
                for translation in translations:
                    f.write(f"{translation}\n")



def main(hparams=None):
    # Load command line hyperparameters (and if provided from an hparams_file).
    if hparams is None:
        if "--hparams_file" not in sys.argv:
            # TODO This is added to prevent incorrect overriding of arguments, see Issue #14
            # When resolved, hparams.update_from_file can be used instead.
            output_dir = Path(sys.argv[sys.argv.index("--output_dir") + 1])
            hparams_file = str(output_dir / "hparams")
            sys.argv = [sys.argv[0]] + ['--hparams_file', hparams_file] + sys.argv[1:]
        hparams = Hyperparameters(check_required=False)

    engine = GenerationEngine(hparams)

    engine.load_statics()

    #if hparams.translation.interactive > 0:
    #    if hparams.translation.interactive == 1:
    #        engine.interactive_translation()
    #    else:
    #        engine.interactive_translation_n(wait_for=hparams.translation.interactive)
    #elif hparams.translation.input_file == '-':
    #    engine.translate_stdin()
    #else:
    #    if hparams.translation.ref_file and hparams.split_sentences:
    #        raise ValueError("If you enable sentence splitting you will compromise line-alignment with the reference")
    #    engine.translate_file(
    #        input_path=hparams.translation.input_file,
    #        output_path=hparams.translation.output_file,
    #        reference_path=hparams.translation.ref_file
    #    )
    engine.generate_file(output_path=hparams.translation.output_file)

if __name__ == "__main__":
    main()
