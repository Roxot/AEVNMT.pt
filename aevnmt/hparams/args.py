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

# Format: "option_name": (type, default_val, required, description, group)
# `group` is for ordering purposes when printing.

io_args = {
    "training_prefix": (str, None, True, "The prefix to bilingual training data."),
    "validation_prefix": (str, None, True, "The validation file prefix."),
    "mono_src": (str, None, False, "The source monolingual training data."),
    "mono_tgt": (str, None, False, "The target monolingual training data."),

    "src": (str, None, True, "The source language"),
    "tgt": (str, None, True, "The target language"),
    "use_gpu": (bool, False, False, "Whether to use the GPU or not"),
    "use_memmap": (bool, False, False, "Whether or not to memory-map the data (use it for large corpora)"),
    "output_dir": (str, None, True, "Output directory."),
    "subword_token": (str, None, False, "The subword token, e.g. \"@@\"."),
    "max_sentence_length": (int, -1, False, "The maximum sentence length during"
                                            " training."),
    "vocab.prefix": (str, None, False, "The vocabulary prefix, if share_vocab is True"
                                       " this should be the vocabulary filename."),
    "vocab.shared": (bool, False, False, "Whether to share the vocabulary between the"
                                       " source and target language"),
    "vocab.max_size": (int, -1, False, "The maximum vocabulary size."),
    "vocab.min_freq": (int, 0, False, "The minimum frequency of a word for it"
                                      " to be included in the vocabulary."),
    "model.checkpoint": (str, None, False, "Checkpoint to restore the model from at"
                                        " the beginning of training."),
}

model_args = {
    # General model hyperparameters.
    "model.type": (str, "cond_nmt", False, "The type of model to train:"
                                           " cond_nmt|aevnmt"),
}

distribution_args = {
    "prior.family": (str, "gaussian", False, "Choose the prior family (gaussian: default, beta, mog)"),
    "prior.params": (str, [], False, "Prior parameters: gaussian (loc: default 0.0, scale: default 1.0), "
        "beta (a: default 0.5, b: default 0.5), "
        "mog (num_components: default 10, radius: default 10, scale: default 0.5)"),
    "prior.latent_size": (int, 32, False, "The size of the latent variables."),
    "prior.latent_sizes": (str, "", False, "Use this to specify latent_size for each prior should you have multiple priors. Example '64;12' "),
    "posterior.family": (str, "gaussian", False, "Choose the family of the approximate posterior (gaussian, kumaraswamy)"),
}

emb_args = {
    "emb.shared": (bool, False, False, "Inf model borrows src embeddings from gen model"),
    "emb.init_scale": (float, 0.01, False, "Scale of the Gaussian that is used to"
                                           " initialize the embeddings."),
    "emb.size": (int, 32, False, "The source / target embedding size, this is also"
                                 " the model size in the transformer architecture.")
}

inf_args = {
    "inf.transformer.hidden_size": (int, 2048, False, "The size of the feedforward layer in the inf transformer"),# TODO
    "inf.transformer.num_layers": (int, 4, False, "The number of inf transformer layers"),# TODO
    "inf.transformer.num_heads": (int, 8, False, "The number of inf transformer heads"),# TODO
    "inf.rnn.hidden_size": (int, 32, False, "The size of the inf rnn layers"),# TODO
    "inf.rnn.num_layers": (int, 1, False, "The number of inf rnn layers."),# TODO
    "inf.rnn.cell_type": (str, "lstm", False, "The inf rnn cell type. rnn|gru|lstm"),# TODO
    "inf.rnn.bidirectional": (bool, False, False, "Use bidirectional inf rnn"),# TODO
    "inf.composition": (str, "avg", False, "Type of composition used after the inference network. avg|maxpool"),# TODO
    "inf.inf3_comb_composition": (str, "cat", False, "Composition function used to combined encodings for q(z|x,y) if --inf3 is set"),
    "inf.inf3": (str, "", False, "Specify encoders for three different inference models, namely, q(z|x), q(z|y) and q(z|x,y),"
                                 "e.g. rnn,rnn,nli or rnn,rnn,comb. The special type 'comb' uses the other two encoders to make an encoding of the pair."),
    "inf.conditioning": (str, "x", False, "Conditioning context for q(z): x|xy"),
    "inf.style": (str, "rnn", False, "The type of inference architecture: rnn|nli|transformer")
}

tm_args = {
    # NOTE: For now encoder and decoder share the same transformer/rnn params, 
    # since many configurations with split params are not supported.
    "gen.tm.transformer.hidden_size": (int, 2048, False, "The size of the feedforward layer in the tm transformer"), # TODO
    "gen.tm.transformer.num_layers": (int, 4, False, "The number of tm transformer layers"), # TODO
    "gen.tm.transformer.num_heads": (int, 8, False, "The number of tm transformer heads"), # TODO
    "gen.tm.rnn.hidden_size": (int, 32, False, "The size of the tm rnn layers"), # TODO
    "gen.tm.rnn.num_layers": (int, 1, False, "The number of tm rnn layers."), # TODO
    "gen.tm.rnn.cell_type": (str, "lstm", False, "The tm rnn cell type. rnn|gru|lstm"), # TODO
    "gen.tm.rnn.attention": (str, "luong", False, "Attention type: luong|scaled_luong|bahdanau"),
    "gen.tm.rnn.bidirectional": (bool, False, False, "Use a bidirectional tm encoder."), # TODO
    "gen.tm.dec.feed_z": (bool, False, False, "Concatenate z to the previous tm decoder embeddings at each timestep"),# TODO
    "gen.tm.dec.tied_embeddings": (bool, False, False, "Tie the tm decoder embedding matrix with the output projection"),# TODO
    "gen.tm.dec.style": (str, "luong", False, "TM decoder style: luong|bahdanau|transformer"),# TODO
    "gen.tm.enc.style": (str, "rnn", False, "TM encoder style: rnn|transformer"),# TODO
}

lm_args = {
    "gen.lm.transformer.hidden_size": (int, 2048, False, "The size of the feedforward layer in the lm transformer"), # TODO
    "gen.lm.transformer.num_layers": (int, 4, False, "The number of lm transformer layers"), # TODO
    "gen.lm.transformer.num_heads": (int, 8, False, "The number of lm transformer heads"), # TODO
    "gen.lm.rnn.hidden_size": (int, 32, False, "The size of the lm rnn layers"), # TODO
    "gen.lm.rnn.num_layers": (int, 1, False, "The number of lm rnn layers."), # TODO
    "gen.lm.rnn.cell_type": (str, "lstm", False, "The lm rnn cell type. rnn|gru|lstm"), # TODO
    "gen.lm.tied_embeddings": (bool, False, False, "Tie the lm embedding matrix with the output projection"),
    "gen.lm.feed_z": (bool, False, False, "Concatenate z to the previous lm embeddings at each timestep"),
    "gen.lm.style": (str, "rnn", False, "LM style: rnn|transformer"),    
}

aux_args = {
    "aux.hidden_size": (int, 32, False, "The size of hidden layers in aux modules"),
    "aux.bow":(bool, False, False, "Add SL bag-of-words term to the loss"),
    "aux.bow_tl":(bool, False, False, "Add TL bag-of-words term to the loss"),
    "aux.MADE":(bool, False, False, "Add SL MADE term to the loss"),
    "aux.MADE_tl":(bool, False, False, "Add TL MADE term to the loss"),
    "aux.count_MADE":(bool, False, False, "Add SL count MADE term to the loss"),
    "aux.count_MADE_tl":(bool, False, False, "Add TL count MADE term to the loss"),
    "aux.ibm1":(bool, False, False, "Side loss based on IBM1-style likelihood p(y|x,z)"),
    "aux.shuffle_lm":(bool, False, False, "z is also used to produce source sentences with a shuffled LM instead of a reverse LM"),
    "aux.shuffle_lm_tl":(bool, False, False, "z is also used to produce target sentences with a shuffled LM instead of a reverse LM"),
    "aux.shuffle_lm_keep_bpe":(bool, False, False, "Shuffle whole words instead of BPE fragments."),
    "likelihood.mixture":(bool, False, False, "Use a mixture of likelihoods"),
    "likelihood.mixture_dir_prior":(float, 0., False, "Specify a symmetric Dirichlet prior over mixture weights (use 0 for uniform and deterministic weights).")
}

opt_args = {
    # General
    "num_epochs": (int, 1, False, "The number of epochs to train the model for."),
    "batch_size": (int, 64, False, "The batch size."),
    "print_every": (int, 100, False, "Print training statistics every x steps."),
    "max_gradient_norm": (float, 5.0, False, "The maximum gradient norm to clip the"
                                             " gradients to, to disable"
                                             " set <= 0."),
    "patience": (int, 5, False, "The number of evaluations to continue training for"
                                " when an improvement has been found."),
    "dropout": (float, 0., False, "The amount of dropout."),
    "word_dropout": (float, 0., False, "Fraction of input words to drop."),
    "evaluate_every": (int, -1, False, "The number of batches after which to run"
                                       " evaluation. If <= 0, evaluation will happen"
                                       " after every epoch."),
    "criterion": (str, "bleu", False, "Criterion for convergence checks ('bleu' or 'likelihood')"),
    # Inf
    "inf.opt.lr_warmup": (int, 4000, False, "Inf learning rate warmup (noam_scheduler)"),# TODO
    "inf.opt.lr_min": (float, 1e-5, False, "The minimum  inf learning rate the learning rate"
                                   " scheduler can reduce to (reduce_on_plateau scheduler)."),# TODO
    "inf.opt.lr_reduce_cooldown": (int, 2, False, "The number of evaluations to wait with"
                                          " checking for improvements after a"
                                          " learning rate reduction on inf"
                                          " (reduce_on_plateau scheduler)."),# TODO
    "inf.opt.lr_reduce_patience": (int, 2, False, "The number of evaluations to wait for"
                                           " improvement of validation scores"
                                           " before reducing the learning rate"
                                           " (reduce_on_plateau scheduler)."),# TODO
    "inf.opt.lr_reduce_factor": (float, 0.5, False, "The factor to reduce the inf learning rate"
                                            " with if no validation improvement is"
                                            " found."), # TODO
    "inf.opt.lr_scheduler": (str, "reduce_on_plateau", False, "The inf learning rate scheduler used: reduce_on_plateau |"
                                                      " noam (transformers)"), # TODO
    "inf.opt.lr": (float, 1e-3, False, "The learning rate for inf_z_optimizer."),
    "inf.opt.l2_weight": (float, 0., False, "Strength of L2 regulariser for inference parameters wrt z"),
    "inf.opt.style": (str, "adam", False, "Optimizer for inference parameters wrt z (options: adam, amsgrad, adadelta, sgd)"),
    # Gen
    "gen.opt.lr_warmup": (int, 4000, False, "Ge learning rate warmup (noam_scheduler)"),# TODO
    "gen.opt.lr_min": (float, 1e-5, False, "The minimum  gen learning rate the learning rate"
                                   " scheduler can reduce to (reduce_on_plateau scheduler)."),# TODO
    "gen.opt.lr_reduce_cooldown": (int, 2, False, "The number of evaluations to wait with"
                                          " checking for improvements after a"
                                          " learning rate reduction on gen"
                                          " (reduce_on_plateau scheduler)."),# TODO
    "gen.opt.lr_reduce_patience": (int, 2, False, "The number of evaluations to wait for"
                                           " improvement of validation scores"
                                           " before reducing the gen learning rate"
                                           " (reduce_on_plateau scheduler)."),# TODO
    "gen.opt.lr_reduce_factor": (float, 0.5, False, "The factor to reduce the gen learning rate"
                                            " with if no validation improvement is"
                                            " found."), # TODO
    "gen.opt.lr_scheduler": (str, "reduce_on_plateau", False, "The gen learning rate scheduler used: reduce_on_plateau |"
                                                      " noam (transformers)"), # TODO
    "gen.opt.lr": (float, 1e-3, False, "The learning rate for gen_z_optimizer."),
    "gen.opt.l2_weight": (float, 0., False, "Strength of L2 regulariser for generative parameters"),
    "gen.opt.style": (str, "adam", False, "Optimizer for generative parameters (options: adam, amsgrad, adadelta, sgd)"),
}

decoding_args = {
    "decoding.sample": (bool, False, False, "When decoding, sample instead of searching for the translation with maximum probability."),
    "decoding.length_penalty_factor": (float, 1.0, False, "Length penalty factor (alpha) for beam search decoding."),
    "decoding.beam_width": (int, 1, False, "Beam search beam width, if 1 this becomes simple greedy decoding"),
    "decoding.max_length": (int, 50, False, "Maximum decoding length"),
}

kl_args = {
    "kl.free_nats": (float, 0., False, "KL = min(KL_free_nats, KL)"),
    "kl.annealing_steps": (int, 0, False, "Amount of KL annealing steps (0...1)"),
    "kl.mdr": (float, -1., False, "If positive adds a soft Lagrangian constraint for the KL term to minimally achieve the given value."),
}

translation_args = {
    # Translation hyperparameters.
    "translation.input_file": (str, None, False, "The translation input file,"
                                               " ignored for training."),
    "translation.output_file": (str, None, False, "The translation output file,"
                                                " ignored for training."),
    "translation.ref_file": (str, None, False, "The translation references file"),
    "verbose": (bool, False, False, "Print logging information"),
    "show_raw_output": (bool, False, False, "Prints raw output (tokenized, truecased, BPE-segmented, max-len splitting) to stderr"),
    "translation.interactive": (int, 0, False, "If n more than 0, reads n sentences from stdin and translates them to stdout"),
    "split_sentences": (bool, False, False, "Pass the whole input through a sentence splitter (mosestokenizer.MosesSentenceSplitter)"),
    "tokenize": (bool, False, False, "Tokenize input (with sacremoses.MosesTokenizer)"),
    "detokenize": (bool, False, False, "Detokenize output (with sacremoses.MosesDetokenizer)"),
    "lowercase": (bool, False, False, "Lowercase the input"),
    "recase": (bool, False, False, "Recase the output (with sacremoses.Detruecaser)"),
    "truecaser_prefix": (str, None, False, "Truecase and de-truecases using a trained model (with sacremoses.MosesTruecaser) -- slow to load"),
    "bpe.codes_prefix": (str, None, False, "Enable BPE-segmentation by providing a prefix to BPE codes (AEVNMT.pt will add .src and .tgt)"),
    "bpe.merge": (bool, True, False, "Merge subwords via regex"),
    "postprocess_ref": (bool, False, False, "Applies post-processing steps to reference (if provided)"),
    "draw_translations": (int, 0, False, "Greedy decode a number of posterior samples"),
}

arg_groups = {
    "IO":  io_args,
    "Model": model_args,
    "Distributions": distribution_args,
    "Embeddings": emb_args,
    "Inference": inf_args,
    "Translation Model": tm_args,
    "Language Model": lm_args,
    "Aux Losses": aux_args,
    "Optimization": opt_args,
    "Decoding": decoding_args,
    "KL": kl_args,
    "Translation": translation_args
}

all_args = {k: v for group in arg_groups.values() for k, v in group.items()}
