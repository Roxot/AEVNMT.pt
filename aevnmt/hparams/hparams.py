import argparse
import json

options = {
    # Format: "option_name": (type, default_val, required, description, group)
    # `group` is for ordering purposes when printing.

    # I/O and device information
    "hparams_file": (str, None, False, "A JSON file containing hyperparameter values.", 0),
    "training_prefix": (str, None, True, "The prefix to bilingual training data.", 0),
    "validation_prefix": (str, None, True, "The validation file prefix.", 0),
    "mono_src": (str, None, False, "The source monolingual training data.", 0),
    "mono_tgt": (str, None, False, "The target monolingual training data.", 0),
    "vocab_prefix": (str, None, False, "The vocabulary prefix, if share_vocab is True"
                                       " this should be the vocabulary filename.", 0),
    "share_vocab": (bool, False, False, "Whether to share the vocabulary between the"
                                       " source and target language", 0),
    "src": (str, None, True, "The source language", 0),
    "tgt": (str, None, True, "The target language", 0),
    "use_gpu": (bool, False, False, "Whether to use the GPU or not", 0),
    "use_memmap": (bool, False, False, "Whether or not to memory-map the data (use it for large corpora)", 0),
    "output_dir": (str, None, True, "Output directory.", 0),
    "subword_token": (str, None, False, "The subword token, e.g. \"@@\".", 0),
    "max_sentence_length": (int, -1, False, "The maximum sentence length during"
                                            " training.", 0),
    "max_vocab_size": (int, -1, False, "The maximum vocabulary size.", 0),
    "vocab_min_freq": (int, 0, False, "The minimum frequency of a word for it"
                                      " to be included in the vocabulary.", 0),
    "model_checkpoint": (str, None, False, "Checkpoint to restore the model from at"
                                           " the beginning of training.", 0),

    # Model hyperparameters
    "model_type": (str, "cond_nmt", False, "The type of model to train:"
                                           " aevnmt", 1),
    "cell_type": (str, "lstm", False, "The RNN cell type. rnn|gru|lstm", 1),
    "emb_size": (int, 32, False, "The source / target embedding size.", 1),
    "hidden_size": (int, 32, False, "The size of the hidden layers.", 1),
    "latent_size": (int, 32, False, "The size of the latent variables.", 1),
    "num_enc_layers": (int, 1, False, "The number of encoder layers.", 1),
    "num_dec_layers": (int, 1, False, "The number of decoder layers.", 1),
    "bidirectional": (bool, False, False, "Use a bidirectional encoder.", 1),
    "attention": (str, "luong", False, "Attention type: luong|scaled_luong|bahdanau", 1),
    "decoder_style": (str, "luong", False, "Decoder style: luong|bahdanau", 1),
    "emb_init_scale": (float, 0.01, False, "Scale of the Gaussian that is used to"
                                           " initialize the embeddings.", 1),
    "tied_embeddings": (bool, False, False, "Tie the embedding matrix with the output"
                                            " projection.", 1),

    # Decoding hyperparameters.
    "max_decoding_length": (int, 50, False, "Maximum decoding length", 2),
    "beam_width": (int, 1, False, "Beam search beam width, if 1 this becomes simple"
                                  " greedy decoding", 2),
    "length_penalty_factor": (float, 1.0, False, "Length penalty factor (alpha) for"
                                                 " beam search decoding.", 2),
    "sample_decoding": (bool, False, False, "When decoding, sample instead of searching for the translation with maximum probability.", 2),

    # Optimization hyperparameters
    "gen_optimizer": (str, "adam", False, "Optimizer for generative parameters (options: adam, amsgrad, adadelta, sgd)", 3),
    "gen_lr": (float, 1e-3, False, "The learning rate for gen_optimizer.", 3),
    "gen_l2_weight": (float, 0., False, "Strength of L2 regulariser for generative parameters", 3),

    "inf_z_optimizer": (str, "adam", False, "Optimizer for inference parameters wrt z (options: adam, amsgrad, adadelta, sgd)", 3),
    "inf_z_lr": (float, 1e-3, False, "The learning rate for inf_z_optimizer.", 3),
    "inf_z_l2_weight": (float, 0., False, "Strength of L2 regulariser for inference parameters wrt z", 3),

    "num_epochs": (int, 1, False, "The number of epochs to train the model for.", 3),
    "batch_size": (int, 64, False, "The batch size.", 3),
    "print_every": (int, 100, False, "Print training statistics every x steps.", 3),
    "max_gradient_norm": (float, 5.0, False, "The maximum gradient norm to clip the"
                                             " gradients to, to disable"
                                             " set <= 0.", 3),
    "lr_reduce_factor": (float, 0.5, False, "The factor to reduce the learning rate"
                                            " with if no validation improvement is"
                                            "  found.", 3),
    "lr_reduce_patience": (int, 2, False, "The number of evaluations to wait for"
                                           " improvement of validation scores"
                                           " before reducing the learning rate.", 3),
    "lr_reduce_cooldown": (int, 2, False, "The number of evaluations to wait with"
                                          " checking for improvements after a"
                                          " learning rate reduction.", 3),
    "min_lr": (float, 1e-5, False, "The minimum learning rate the learning rate"
                                   " scheduler can reduce to.", 3),
    "patience": (int, 5, False, "The number of evaluations to continue training for"
                                " when an improvement has been found.", 3),
    "dropout": (float, 0., False, "The amount of dropout.", 3),
    "word_dropout": (float, 0., False, "Fraction of input words to drop.", 3),
    "KL_free_nats": (float, 0., False, "KL = min(KL_free_nats, KL)", 3),
    "KL_annealing_steps": (int, 0, False, "Amount of KL annealing steps (0...1)", 3),
    "evaluate_every": (int, -1, False, "The number of batches after which to run"
                                       " evaluation. If <= 0, evaluation will happen"
                                       " after every epoch.", 3),
    "criterion": (str, "bleu", False, "Criterion for convergence checks ('bleu' or 'likelihood')", 3),

    # Translation hyperparameters.
    "translation_input_file": (str, None, False, "The translation input file,"
                                               " ignored for training.", 4),
    "translation_output_file": (str, None, False, "The translation output file,"
                                                " ignored for training.", 4),
    "translation_ref_file": (str, None, False, "The translation references file", 4),
    "verbose": (bool, False, False, "Print logging information", 4),
    "show_raw_output": (bool, False, False, "Prints raw output (tokenized, truecased, BPE-segmented, max-len splitting) to stderr", 4),
    "interactive_translation": (int, 0, False, "If n more than 0, reads n sentences from stdin and translates them to stdout", 4),
    "split_sentences": (bool, False, False, "Pass the whole input through a sentence splitter (mosestokenizer.MosesSentenceSplitter)", 4),
    "tokenize": (bool, False, False, "Tokenize input (with sacremoses.MosesTokenizer)", 4),
    "detokenize": (bool, False, False, "Detokenize output (with sacremoses.MosesDetokenizer)", 4),
    "lowercase": (bool, False, False, "Lowercase the input", 4),
    "recase": (bool, False, False, "Recase the output (with sacremoses.Detruecaser)", 4),
    "truecaser_prefix": (str, None, False, "Truecase and de-truecases using a trained model (with sacremoses.MosesTruecaser) -- slow to load", 4),
    "bpe_codes_prefix": (str, None, False, "Enable BPE-segmentation by providing a prefix to BPE codes (AEVNMT.pt will add .src and .tgt)", 4),
    "bpe_merge": (bool, True, False, "Merge subwords via regex", 4),
    "postprocess_ref": (bool, False, False, "Applies post-processing steps to reference (if provided)", 4),

    "draw_translations": (int, 0, False, "Greedy decode a number of posterior samples", 5),
}

class Hyperparameters:

    """
        Loads hyperparameters from the command line arguments and optionally
        from a JSON file. Command line arguments overwrite those from the JSON file.
    """
    def __init__(self, check_required=True):
        self._hparams = {}
        self._defaulted_values = []
        cmd_line_hparams = self._load_from_command_line()

        # If given, load hparams from a json file.
        json_file = cmd_line_hparams["hparams_file"] if "hparams_file" in cmd_line_hparams \
                else None
        if json_file is not None:
            json_hparams = self._load_from_json(json_file)
            self._hparams.update(json_hparams)

        # Always override json hparams with command line hparams.
        self._hparams.update(cmd_line_hparams)

        # Set default values, check for required, set hparams as attributes.
        self._create_hparams(check_required)

    def update_from_file(self, json_file, override=False):
        json_hparams = self._load_from_json(json_file)
        if not override:
            for key, val in list(json_hparams.items()):
                if key not in self._defaulted_values:
                    del json_hparams[key]
                else:
                    self._defaulted_values.remove(key)
        self._hparams.update(json_hparams)
        self._create_hparams(False)

    def update_from_kwargs(self, **kwargs):
        self._hparams.update(kwargs)
        self._create_hparams(False)

    def _load_from_command_line(self):
        parser = argparse.ArgumentParser()
        for option in options.keys():
            option_type, _, _, description, _ = options[option]
            option_type = str if option_type == bool else option_type
            parser.add_argument(f"--{option}", type=option_type,
                                help=description)
        args = parser.parse_known_args()[0]
        cmd_line_hparams = vars(args)
        for key, val in list(cmd_line_hparams.items()):
            if val is None:
                del cmd_line_hparams[key]
        return cmd_line_hparams

    def _load_from_json(self, filename):
        with open(filename) as f:
            json_hparams = json.load(f)
        return json_hparams

    def _create_hparams(self, check_required):
        for option in options.keys():
            option_type, default_value, required, _, _ = options[option]
            if option not in self._hparams:
                self._hparams[option] = default_value
                self._defaulted_values.append(option)
            elif option_type == bool and isinstance(self._hparams[option], str):

                # Convert boolean inputs from string to bool. Only necessary if the
                # default value is not used.
                self._hparams[option] = str_to_bool(self._hparams[option])

            if self._hparams[option] is None:

                # Raise an error if required.
                if check_required and required:
                    raise Exception(f"Error: missing required value `{option}`.")

            setattr(self, option, self._hparams[option])

    def print_values(self):
        sorted_names = sorted(options.keys(), key=lambda name: (options[name][-1], name))
        cur_group = 0
        for name in sorted_names:
            if options[name][-1] > cur_group:
                print()
                cur_group = options[name][-1]

            val = self._hparams[name]
            defaulted = "(default)" if name in self._defaulted_values else ""
            print(f"{name} = {val} {defaulted}")

    def save(self, filename):
        with open(filename, "w") as f:
            json.dump(self._hparams, f, sort_keys=True, indent=4)

def str_to_bool(string):
    return string.lower() == "true"
