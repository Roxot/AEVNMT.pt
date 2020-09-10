from argparse import ArgumentTypeError


def str_to_str_list(values):
    if isinstance(values, str):
        values = (str(v) for v in values.split())
    else:
        values = (str(v) for v in values)
    return list(values)


def str_to_float_list(values):
    if isinstance(values, str):
        values = (float(v) for v in values.split())
    else:
        values = (float(v) for v in values)
    return list(values)


def str_to_int_list(values):
    if isinstance(values, str):
        values = (int(v) for v in values.split())
    else:
        values = (int(v) for v in values)
    return list(values)


def str_to_bool(v):
    """Source: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise ArgumentTypeError('Boolean value expected.')


io_args = {
    "training_prefix": (str, None, True, "The prefix to bilingual training data.", 0),
    "validation_prefix": (str, None, True, "The validation file prefix.", 0),
    "mono_src": (str, None, False, "The source monolingual training data.", 0),
    "mono_tgt": (str, None, False, "The target monolingual training data.", 0),

    "src": (str, None, True, "The source language", 0),
    "tgt": (str, None, True, "The target language", 0),
    "use_gpu": (bool, False, False, "Whether to use the GPU or not", 0),
    "use_memmap": (bool, False, False, "Whether or not to memory-map the data (use it for large corpora)", 0),
    "output_dir": (str, None, True, "Output directory.", 0),
    "subword_token": (str, None, False, "The subword token, e.g. \"@@\".", 0),
    "max_sentence_length": (int, -1, False, "The maximum sentence length during"
                                            " training.", 0),
    "vocab.prefix": (str, None, False, "The vocabulary prefix, if share_vocab is True"
                                       " this should be the vocabulary filename.", 0),
    "vocab.shared": (bool, False, False, "Whether to share the vocabulary between the"
                                       " source and target language", 0),
    "vocab.max_size": (int, -1, False, "The maximum vocabulary size.", 0),
    "vocab.min_freq": (int, 0, False, "The minimum frequency of a word for it"
                                      " to be included in the vocabulary.", 0),
    "model.checkpoint": (str, None, False, "Checkpoint to restore the model from at"
                                        " the beginning of training.", 0),
}

model_args = {
    # Format: "option_name": (type, default_val, required, description, group)
    # `group` is for ordering purposes when printing.

    # General model hyperparameters.
    "model.type": (str, "cond_nmt", False, "The type of model to train:"
                                           " cond_nmt|aevnmt", 1),
    "prior.type": (str, "gaussian", False, "Choose the prior family (gaussian: default, beta, mog)", 1),
    "prior.params": (str, [], False, "Prior parameters: gaussian (loc: default 0.0, scale: default 1.0), "
        "beta (a: default 0.5, b: default 0.5), "
        "mog (num_components: default 10, radius: default 10, scale: default 0.5)", 2),
    "latent.size": (int, 32, False, "The size of the latent variables.", 1),
    "latent.sizes": (str, "", False, "Use this to specify latent_size for each prior should you have multiple priors. Example '64;12' ", 1),
    "emb.size": (int, 32, False, "The source / target embedding size, this is also"
                                 " the model size in the transformer architecture.", 1),
    "emb.init_scale": (float, 0.01, False, "Scale of the Gaussian that is used to"
                                           " initialize the embeddings.", 1),
    "hidden.size": (int, 32, False, "The size of the hidden layers.", 1),
    "emb.tied": (bool, False, False, "Tie the embedding matrix with the output"
                                            " projection.", 1),
    "max_pooling_states":(bool, False, False, "Max-pool encoder states in the inference network instead of averaging them", 1),
    "feed_z":(bool, False, False, "Concatenate z to the previous word embeddings at each timestep", 1),

    "enc.style": (str, "rnn", False, "The type of encoder architecture: rnn|transformer", 1),
    "dec.style": (str, "luong", False, "Decoder style: luong|bahdanau", 1),
    "enc.num_layers": (int, 1, False, "The number of encoder layers.", 1),
    "dec.num_layers": (int, 1, False, "The number of decoder layers.", 1),
}

aux_args = {
    "aux.bow":(bool, False, False, "Add SL bag-of-words term to the loss", 1),
    "aux.bow_tl":(bool, False, False, "Add TL bag-of-words term to the loss", 1),
    "aux.MADE":(bool, False, False, "Add SL MADE term to the loss", 1),
    "aux.MADE_tl":(bool, False, False, "Add TL MADE term to the loss", 1),
    "aux.count_MADE":(bool, False, False, "Add SL count MADE term to the loss", 1),
    "aux.count_MADE_tl":(bool, False, False, "Add TL count MADE term to the loss", 1),
    "aux.ibm1":(bool, False, False, "Side loss based on IBM1-style likelihood p(y|x,z)", 1),
    "aux.shuffle_lm":(bool, False, False, "z is also used to produce source sentences with a shuffled LM instead of a reverse LM", 1),
    "aux.shuffle_lm_tl":(bool, False, False, "z is also used to produce target sentences with a shuffled LM instead of a reverse LM", 1),
    "aux.shuffle_lm_keep_bpe":(bool, False, False, "Shuffle whole words instead of BPE fragments.", 1),
    "likelihood.mixture":(bool, False, False, "Use a mixture of likelihoods", 1),
    "likelihood.mixture_dir_prior":(float, 0., False, "Specify a symmetric Dirichlet prior over mixture weights (use 0 for uniform and deterministic weights).", 1)
}

transformer_args = {
    "transformer.heads": (int, 8, False, "The number of transformer heads in that architecture", 1),
    "transformer.hidden": (int, 2048, False, "The size of the hidden feedforward layer in the transformer", 1)
}

rnn_args = {
    # RNN encoder / decoder hyperparameters.
    "bidirectional": (bool, False, False, "Use a bidirectional encoder.", 1),
    "cell_type": (str, "lstm", False, "The RNN cell type. rnn|gru|lstm", 1),
    "attention": (str, "luong", False, "Attention type: luong|scaled_luong|bahdanau", 1)
}

inf_args = {
    # Inference model hyperparameters
    "posterior.type": (str, "gaussian", False, "Choose the family of the posterior approximation (gaussian, kumaraswamy)", 2),
    "inf.encoder_style": (str, "rnn", False, "The type of architecture: rnn|nli", 2),
    "inf.conditioning": (str, "x", False, "Conditioning context for q(z): x|xy", 2),
    "inf.share_embeddings": (bool, False, False, "Should the inference model borrow embeddings from generative model?", 2),
    "inf3": (str, "", False, "Specify encoders for three different inference models, namely, q(z|x), q(z|y) and q(z|x,y), e.g. rnn,rnn,nli or rnn,rnn,comb. The special type 'comb' uses the other two encoders to make an encoding of the pair.", 2),  
    "inf3_comb_composition": (str, "cat", False, "Composition function used to combined encodings for q(z|x,y) if --inf3 is set", 2),

}

dec_args = {
    # Decoding hyperparameters.
    "dec.max_length": (int, 50, False, "Maximum decoding length", 3),
    "beam_width": (int, 1, False, "Beam search beam width, if 1 this becomes simple"
                                  " greedy decoding", 3),
    "length_penalty_factor": (float, 1.0, False, "Length penalty factor (alpha) for"
                                                 " beam search decoding.", 3),
    "dec.sample": (bool, False, False, "When decoding, sample instead of searching for the translation with maximum probability.", 3),
}

opt_args = {
    # Optimization hyperparameters
    "gen.opt": (str, "adam", False, "Optimizer for generative parameters (options: adam, amsgrad, adadelta, sgd)", 4),
    "gen.lr": (float, 1e-3, False, "The learning rate for gen_optimizer.", 4),
    "gen.l2_weight": (float, 0., False, "Strength of L2 regulariser for generative parameters", 4),

    "inf.opt": (str, "adam", False, "Optimizer for inference parameters wrt z (options: adam, amsgrad, adadelta, sgd)", 4),
    "inf.lr": (float, 1e-3, False, "The learning rate for inf_z_optimizer.", 4),
    "inf.l2_weight": (float, 0., False, "Strength of L2 regulariser for inference parameters wrt z", 4),

    "num_epochs": (int, 1, False, "The number of epochs to train the model for.", 4),
    "batch_size": (int, 64, False, "The batch size.", 4),
    "print_every": (int, 100, False, "Print training statistics every x steps.", 4),
    "max_gradient_norm": (float, 5.0, False, "The maximum gradient norm to clip the"
                                             " gradients to, to disable"
                                             " set <= 0.", 4),
    "lr.scheduler": (str, "reduce_on_plateau", False, "The learning rate scheduler used: reduce_on_plateau |"
                                                      " noam (transformers)", 4),
    "lr.reduce_factor": (float, 0.5, False, "The factor to reduce the learning rate"
                                            " with if no validation improvement is"
                                            "  found (all lr schedulers).", 4),
    "lr.reduce_patience": (int, 2, False, "The number of evaluations to wait for"
                                           " improvement of validation scores"
                                           " before reducing the learning rate"
                                           " (reduce_on_plateau scheduler).", 4),
    "lr.reduce_cooldown": (int, 2, False, "The number of evaluations to wait with"
                                          " checking for improvements after a"
                                          " learning rate reduction"
                                          " (reduce_on_plateau scheduler).", 4),
    "lr.min": (float, 1e-5, False, "The minimum learning rate the learning rate"
                                   " scheduler can reduce to (reduce_on_plateau scheduler).", 4),
    "lr.warmup": (int, 4000, False, "Learning rate warmup (noam_scheduler)", 4),
    "patience": (int, 5, False, "The number of evaluations to continue training for"
                                " when an improvement has been found.", 4),
    "dropout": (float, 0., False, "The amount of dropout.", 4),
    "word_dropout": (float, 0., False, "Fraction of input words to drop.", 4),
    "evaluate_every": (int, -1, False, "The number of batches after which to run"
                                       " evaluation. If <= 0, evaluation will happen"
                                       " after every epoch.", 4),
    "criterion": (str, "bleu", False, "Criterion for convergence checks ('bleu' or 'likelihood')", 4)
}

kl_args = {
    "kl.free_nats": (float, 0., False, "KL = min(KL_free_nats, KL)", 4),
    "kl.annealing_steps": (int, 0, False, "Amount of KL annealing steps (0...1)", 4),
    "kl.mdr": (float, -1., False, "If positive adds a soft Lagrangian constraint for the KL term to"
                                                " minimally achieve the given value.", 4),
}

translation_args = {
    # Translation hyperparameters.
    "translation.input_file": (str, None, False, "The translation input file,"
                                               " ignored for training.", 5),
    "translation.output_file": (str, None, False, "The translation output file,"
                                                " ignored for training.", 5),
    "translation.ref_file": (str, None, False, "The translation references file", 5),
    "verbose": (bool, False, False, "Print logging information", 5),
    "show_raw_output": (bool, False, False, "Prints raw output (tokenized, truecased, BPE-segmented, max-len splitting) to stderr", 5),
    "translation.interactive": (int, 0, False, "If n more than 0, reads n sentences from stdin and translates them to stdout", 5),
    "split_sentences": (bool, False, False, "Pass the whole input through a sentence splitter (mosestokenizer.MosesSentenceSplitter)", 5),
    "tokenize": (bool, False, False, "Tokenize input (with sacremoses.MosesTokenizer)", 5),
    "detokenize": (bool, False, False, "Detokenize output (with sacremoses.MosesDetokenizer)", 5),
    "lowercase": (bool, False, False, "Lowercase the input", 5),
    "recase": (bool, False, False, "Recase the output (with sacremoses.Detruecaser)", 5),
    "truecaser_prefix": (str, None, False, "Truecase and de-truecases using a trained model (with sacremoses.MosesTruecaser) -- slow to load", 5),
    "bpe.codes_prefix": (str, None, False, "Enable BPE-segmentation by providing a prefix to BPE codes (AEVNMT.pt will add .src and .tgt)", 5),
    "bpe.merge": (bool, True, False, "Merge subwords via regex", 5),
    "postprocess_ref": (bool, False, False, "Applies post-processing steps to reference (if provided)", 5),
    "draw_translations": (int, 0, False, "Greedy decode a number of posterior samples", 5),
}

arg_groups = {
    "IO":  io_args,
    "Model": model_args,
    "Aux Losses": aux_args,
    "RNN": rnn_args,
    "Transformer": transformer_args,
    "Inference": inf_args,
    "Decoding": dec_args,
    "Optimization": opt_args,
    "KL": kl_args,
    "Translation": translation_args
}

all_args = {k: v for group in arg_groups.values() for k, v in group.items()}
