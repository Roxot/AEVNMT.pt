import sys

from jsonargparse import ArgumentParser, ActionConfigFile

CFG_FILE_ARG = "--hparams_file"

def main():
    argv = sys.argv[1:]
    if CFG_FILE_ARG in argv:
        cfg_arg_idx = argv.index('--config')
        argv = argv[cfg_idx:cfg_idx+2] + argv[:cfg_idx] + argv[cfg_idx+2:]

    parser = ArgumentParser()
    parser.add_argument(CFG_FILE_ARG, action=ActionConfigFile)
    add_io_args(parser)
    args = parser.parse_args(argv, env=None, nested=True)
    print(args)

def add_io_args(parser):
    group = parser.add_argument_group("IO")
    group.add_argument("--training_prefix", type=str, default=None,
                       help="The prefix to bilingual training data.")
    group.add_argument("--validation_prefix", type=str, default=None,
                       help="The validation file prefix.")

def add_gen_args(parser):
    parser.add_argument("--prior", action="append")

if __name__ == '__main__':
    main()

"""
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

    # General model hyperparameters.
    "model_type": (str, "cond_nmt", False, "The type of model to train:"
                                           " cond_nmt|aevnmt", 1),
    "prior": (str, "gaussian", False, "Choose the prior family (gaussian: default, beta, mog)", 1),
    "prior_params": (ListOfFloats, [], False, "Prior parameters: gaussian (loc: default 0.0, scale: default 1.0), "
        "beta (a: default 0.5, b: default 0.5), "
        "mog (num_components: default 10, radius: default 10, scale: default 0.5)", 2),
    "latent_size": (int, 32, False, "The size of the latent variables.", 1),
    "emb_size": (int, 32, False, "The source / target embedding size, this is also"
                                 " the model size in the transformer architecture.", 1),
    "emb_init_scale": (float, 0.01, False, "Scale of the Gaussian that is used to"
                                           " initialize the embeddings.", 1),
    "hidden_size": (int, 32, False, "The size of the hidden layers.", 1),
    "num_enc_layers": (int, 1, False, "The number of encoder layers.", 1),
    "num_dec_layers": (int, 1, False, "The number of decoder layers.", 1),
    "tied_embeddings": (bool, False, False, "Tie the embedding matrix with the output"
                                            " projection.", 1),
    "max_pooling_states":(bool, False, False, "Max-pool encoder states in the inference network instead of averaging them", 1),
    "feed_z":(bool, False, False, "Concatenate z to the previous word embeddings at each timestep", 1),

    "bow_loss":(bool, False, False, "Add SL bag-of-words term to the loss", 1),
    "bow_loss_tl":(bool, False, False, "Add TL bag-of-words term to the loss", 1),
    "MADE_loss":(bool, False, False, "Add SL MADE term to the loss", 1),
    "MADE_loss_tl":(bool, False, False, "Add TL MADE term to the loss", 1),
    "count_MADE_loss":(bool, False, False, "Add SL count MADE term to the loss", 1),
    "count_MADE_loss_tl":(bool, False, False, "Add TL count MADE term to the loss", 1),
    "shuffle_lm":(bool, False, False, "z is also used to produce source sentences with a shuffled LM instead of a reverse LM", 1),
    "shuffle_lm_tl":(bool, False, False, "z is also used to produce target sentences with a shuffled LM instead of a reverse LM", 1),
    "shuffle_lm_keep_bpe":(bool, False, False, "Shuffle whole words instead of BPE fragments.", 1),
    "mixture_likelihood":(bool, False, False, "Use a mixture of likelihoods", 1),
    "mixture_likelihood_dir_prior":(float, 0., False, "Specify a symmetric Dirichlet prior over mixture weights (use 0 for uniform and deterministic weights).", 1),
    
    "ibm1_loss":(bool, False, False, "Side loss based on IBM1-style likelihood p(y|x,z)", 1),

    "encoder_style": (str, "rnn", False, "The type of encoder architecture: rnn|transformer", 1),
    "decoder_style": (str, "luong", False, "Decoder style: luong|bahdanau", 1),

    # Transformer encoder / decoder hyperparameters.
    "transformer_heads": (int, 8, False, "The number of transformer heads in that architecture", 1),
    "transformer_hidden": (int, 2048, False, "The size of the hidden feedforward layer in the transformer", 1),

    # RNN encoder / decoder hyperparameters.
    "bidirectional": (bool, False, False, "Use a bidirectional encoder.", 1),
    "cell_type": (str, "lstm", False, "The RNN cell type. rnn|gru|lstm", 1),
    "attention": (str, "luong", False, "Attention type: luong|scaled_luong|bahdanau", 1),

    # Inference model hyperparameters
    "posterior": (str, "gaussian", False, "Choose the family of the posterior approximation (gaussian, kumaraswamy)", 2),
    "inf_encoder_style": (str, "rnn", False, "The type of architecture: rnn|nli", 2),
    "inf_conditioning": (str, "x", False, "Conditioning context for q(z): x|xy", 2),


    # Decoding hyperparameters.
    "max_decoding_length": (int, 50, False, "Maximum decoding length", 3),
    "beam_width": (int, 1, False, "Beam search beam width, if 1 this becomes simple"
                                  " greedy decoding", 3),
    "length_penalty_factor": (float, 1.0, False, "Length penalty factor (alpha) for"
                                                 " beam search decoding.", 3),
    "sample_decoding": (bool, False, False, "When decoding, sample instead of searching for the translation with maximum probability.", 3),

    # Optimization hyperparameters
    "gen_optimizer": (str, "adam", False, "Optimizer for generative parameters (options: adam, amsgrad, adadelta, sgd)", 4),
    "gen_lr": (float, 1e-3, False, "The learning rate for gen_optimizer.", 4),
    "gen_l2_weight": (float, 0., False, "Strength of L2 regulariser for generative parameters", 4),

    "inf_z_optimizer": (str, "adam", False, "Optimizer for inference parameters wrt z (options: adam, amsgrad, adadelta, sgd)", 4),
    "inf_z_lr": (float, 1e-3, False, "The learning rate for inf_z_optimizer.", 4),
    "inf_z_l2_weight": (float, 0., False, "Strength of L2 regulariser for inference parameters wrt z", 4),

    "num_epochs": (int, 1, False, "The number of epochs to train the model for.", 4),
    "batch_size": (int, 64, False, "The batch size.", 4),
    "print_every": (int, 100, False, "Print training statistics every x steps.", 4),
    "max_gradient_norm": (float, 5.0, False, "The maximum gradient norm to clip the"
                                             " gradients to, to disable"
                                             " set <= 0.", 4),
    "lr_scheduler": (str, "reduce_on_plateau", False, "The learning rate scheduler used: reduce_on_plateau |"
                                                      " noam (transformers)", 4),
    "lr_reduce_factor": (float, 0.5, False, "The factor to reduce the learning rate"
                                            " with if no validation improvement is"
                                            "  found (all lr schedulers).", 4),
    "lr_reduce_patience": (int, 2, False, "The number of evaluations to wait for"
                                           " improvement of validation scores"
                                           " before reducing the learning rate"
                                           " (reduce_on_plateau scheduler).", 4),
    "lr_reduce_cooldown": (int, 2, False, "The number of evaluations to wait with"
                                          " checking for improvements after a"
                                          " learning rate reduction"
                                          " (reduce_on_plateau scheduler).", 4),
    "min_lr": (float, 1e-5, False, "The minimum learning rate the learning rate"
                                   " scheduler can reduce to (reduce_on_plateau scheduler).", 4),
    "lr_warmup": (int, 4000, False, "Learning rate warmup (noam_scheduler)", 4),
    "patience": (int, 5, False, "The number of evaluations to continue training for"
                                " when an improvement has been found.", 4),
    "dropout": (float, 0., False, "The amount of dropout.", 4),
    "word_dropout": (float, 0., False, "Fraction of input words to drop.", 4),
    "KL_free_nats": (float, 0., False, "KL = min(KL_free_nats, KL)", 4),
    "KL_annealing_steps": (int, 0, False, "Amount of KL annealing steps (0...1)", 4),
    "minimum_desired_rate": (float, -1., False, "If positive adds a soft Lagrangian constraint for the KL term to"
                                                " minimally achieve the given value.", 4),
    "evaluate_every": (int, -1, False, "The number of batches after which to run"
                                       " evaluation. If <= 0, evaluation will happen"
                                       " after every epoch.", 4),
    "criterion": (str, "bleu", False, "Criterion for convergence checks ('bleu' or 'likelihood')", 4),

    # Translation hyperparameters.
    "translation_input_file": (str, None, False, "The translation input file,"
                                               " ignored for training.", 5),
    "translation_output_file": (str, None, False, "The translation output file,"
                                                " ignored for training.", 5),
    "translation_ref_file": (str, None, False, "The translation references file", 5),
    "verbose": (bool, False, False, "Print logging information", 5),
    "show_raw_output": (bool, False, False, "Prints raw output (tokenized, truecased, BPE-segmented, max-len splitting) to stderr", 5),
    "interactive_translation": (int, 0, False, "If n more than 0, reads n sentences from stdin and translates them to stdout", 5),
    "split_sentences": (bool, False, False, "Pass the whole input through a sentence splitter (mosestokenizer.MosesSentenceSplitter)", 5),
    "tokenize": (bool, False, False, "Tokenize input (with sacremoses.MosesTokenizer)", 5),
    "detokenize": (bool, False, False, "Detokenize output (with sacremoses.MosesDetokenizer)", 5),
    "lowercase": (bool, False, False, "Lowercase the input", 5),
    "recase": (bool, False, False, "Recase the output (with sacremoses.Detruecaser)", 5),
    "truecaser_prefix": (str, None, False, "Truecase and de-truecases using a trained model (with sacremoses.MosesTruecaser) -- slow to load", 5),
    "bpe_codes_prefix": (str, None, False, "Enable BPE-segmentation by providing a prefix to BPE codes (AEVNMT.pt will add .src and .tgt)", 5),
    "bpe_merge": (bool, True, False, "Merge subwords via regex", 5),
    "postprocess_ref": (bool, False, False, "Applies post-processing steps to reference (if provided)", 5),

    "draw_translations": (int, 0, False, "Greedy decode a number of posterior samples", 5),
}
"""
