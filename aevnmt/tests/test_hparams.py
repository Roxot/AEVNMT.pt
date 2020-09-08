from aevnmt.hparams.nested_hparams import NestedHyperparameters
from aevnmt.hparams.utils import rgetattr
import argparse
import jsonargparse

transformer_args = {
    "transformer.heads": ("transformer.heads", int, 8, False, "The number of transformer heads in that architecture", 1),
    "transformer.hidden": ("transformer.hidden", int, 2048, False, "The size of the hidden feedforward layer in the transformer", 1),
    "opt_test": (str, None, False, "optional argument test", 1)
}

rnn_args = {
    # RNN encoder / decoder hyperparameters.
    "bidirectional": (bool, False, False, "Use a bidirectional encoder.", 1),
    "cell_type": (str, "lstm", False, "The RNN cell type. rnn|gru|lstm", 1),
    "attention": (str, "luong", False, "Attention type: luong|scaled_luong|bahdanau", 1)
}

arg_groups = {
    "RNN": rnn_args,
    "Transformer": transformer_args
}

if __name__ == "__main__":
    hparams = NestedHyperparameters(arg_groups)
    print("DEFAULT CONFIG")
    hparams.print_values()
    print("update, override=False")
    hparams.update_from_file('test_config.yaml', override=False)
    hparams.print_values()
    print("update, override=True")
    hparams.update_from_file('test_config.yaml', override=True)
    hparams.print_values()

