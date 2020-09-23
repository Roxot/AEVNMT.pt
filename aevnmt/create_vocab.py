import sys

from aevnmt.hparams import Hyperparameters
from aevnmt.train_utils import load_vocabularies

from pathlib import Path

def create_vocab():

    # Load and print hyperparameters.
    hparams = Hyperparameters(check_required=True)
    print("\n==== Hyperparameters")
    hparams.print_values()

    # Load the data and print some statistics.
    vocab_src, vocab_tgt = load_vocabularies(hparams)
    if hparams.vocab.shared:
        print("\n==== Vocabulary")
        vocab_src.print_statistics()
    else:
        print("\n==== Source vocabulary")
        vocab_src.print_statistics()
        print("\n==== Target vocabulary")
        vocab_tgt.print_statistics()

    # Create the output directory.
    out_dir = Path(hparams.output_dir)
    if not out_dir.exists():
        out_dir.mkdir()

    print(f"\nSaving vocabularies to {out_dir}...")
    vocab_src.save(out_dir / f"vocab.{hparams.src}")
    vocab_tgt.save(out_dir / f"vocab.{hparams.tgt}")
    hparams.vocab.prefix = str(out_dir / "vocab")

if __name__ == "__main__":
    create_vocab()
