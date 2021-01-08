from collections import defaultdict
from typing import Dict
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, kl_divergence
from torch.utils.data import DataLoader

from probabll.distributions import ProductOfDistributions

from aevnmt.data import BucketingParallelDataLoader,BucketingTextDataLoader, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN
from aevnmt.data import create_batch, batch_to_sentences
from aevnmt.components import DetachedEmbeddingLayer, RNNEncoder, beam_search, greedy_decode, sampling_decode, Constraint, loss_functions
from aevnmt.models import AEVNMT,SenVAE
from aevnmt.models.generative import GenerativeLM, IndependentLM, CorrelatedBernoullisLM
from aevnmt.models.generative import CorrelatedCategoricalsLM, CorrelatedPoissonsLM
from aevnmt.models.generative import GenerativeTM, IndependentTM, CorrelatedBernoullisTM, CorrelatedCategoricalsTM
from aevnmt.models.generative import CorrelatedPoissonsTM, IBM1TM, AttentionBasedTM
from aevnmt.models.generative import TransformerLM, TransformerTM
from aevnmt.models.inference import InferenceModel, BasicInferenceModel, SwitchingInferenceModel
from aevnmt.models.inference import get_inference_encoder, combine_inference_encoders
from aevnmt.models.parallel import ParallelWrapper
from aevnmt.dist import get_named_params, create_prior
from aevnmt.dist import ProductOfPriorsLayer, ProductOfConditionalsLayer
from aevnmt.dist import NormalLayer, KumaraswamyLayer, HardKumaraswamyLayer

from .train_utils import create_attention, create_encoder, create_decoder, attention_summary, compute_bleu


def _draw_translations(model, val_dl, vocab_src, vocab_tgt, device, hparams):
    with torch.no_grad():
        inputs = []
        references = []
        model_hypotheses = []
        for sentences_x, sentences_y in val_dl:
            hypothesis = translate(model, sentences_x, vocab_src, vocab_tgt, device, hparams, deterministic=False)

            # Keep track of inputs, references and model hypotheses.
            inputs += sentences_x.tolist()
            references += sentences_y.tolist()
            model_hypotheses += hypothesis.tolist()
    return inputs, references, model_hypotheses


def create_language_model(vocab_src, src_embedder, hparams) -> GenerativeLM:
    if hparams.gen.lm.style == "rnn":
        language_model = CorrelatedCategoricalsLM(
            embedder=src_embedder,
            sos_idx=vocab_src[SOS_TOKEN],
            eos_idx=vocab_src[EOS_TOKEN],
            latent_size=hparams.prior.latent_size,
            hidden_size=hparams.gen.lm.rnn.hidden_size,
            dropout=hparams.dropout,
            num_layers=hparams.gen.lm.rnn.num_layers,
            cell_type=hparams.gen.lm.rnn.cell_type,
            tied_embeddings=hparams.gen.lm.tied_embeddings,
            feed_z=hparams.gen.lm.feed_z,
            gate_z=False  # TODO implement
        )
    elif hparams.gen.lm.style == "transformer":
        language_model = TransformerLM(
            embedder=src_embedder,
            sos_idx=vocab_src[SOS_TOKEN],
            eos_idx=vocab_src[EOS_TOKEN],
            latent_size=hparams.prior.latent_size,
            hidden_size=hparams.gen.lm.transformer.hidden_size,
            num_heads=hparams.gen.lm.transformer.num_heads,
            num_layers=hparams.gen.lm.transformer.num_layers,
            dropout=hparams.dropout,
            tied_embeddings=hparams.gen.lm.tied_embeddings,
            feed_z_method=["first"] if hparams.gen.lm.feed_z else []
        )
    else:
        raise NotImplementedError(f"Unknown language model style: {hparams.lm.style}")
    return language_model


def create_aux_language_models(vocab_src, src_embedder, hparams) -> Dict[str, GenerativeLM]:
    lms = dict()
    if hparams.aux.bow:
        lms['bow'] = IndependentLM(
            latent_size=hparams.prior.latent_size,
            embedder=src_embedder,
            tied_embeddings=False)
    if hparams.aux.MADE:
        lms['made'] = CorrelatedBernoullisLM(
            vocab_size=src_embedder.num_embeddings,
            latent_size=hparams.prior.latent_size,
            hidden_sizes=[hparams.aux.hidden_size, hparams.aux.hidden_size],  # TODO: generalise
            pad_idx=src_embedder.padding_idx,
            num_masks=1,
            resample_mask_every=0)
    if hparams.aux.count_MADE:
        lms['count_made'] = CorrelatedPoissonsLM(
            vocab_size=src_embedder.num_embeddings,
            latent_size=hparams.prior.latent_size,
            hidden_sizes=[hparams.aux.hidden_size, hparams.aux.hidden_size],  # TODO: generalise
            pad_idx=src_embedder.padding_idx,
            num_masks=1,
            resample_mask_every=0)
    if hparams.aux.shuffle_lm:
        raise NotImplementedError("This is not yet supported")
    return lms


def create_translation_model(vocab_tgt, src_embedder, tgt_embedder, hparams):
    encoder = create_encoder(hparams)
    attention = create_attention(hparams)
    decoder = create_decoder(attention, hparams)

    if hparams.gen.tm.enc.style == 'transformer' and hparams.gen.tm.dec.style == 'transformer':
        translation_model = TransformerTM(
            src_embedder=src_embedder,
            tgt_embedder=tgt_embedder,
            tgt_sos_idx=vocab_tgt[SOS_TOKEN],
            tgt_eos_idx=vocab_tgt[EOS_TOKEN],
            encoder=encoder,
            decoder=decoder,
            latent_size=hparams.prior.latent_size,
            dropout=hparams.dropout,
            tied_embeddings=hparams.gen.tm.dec.tied_embeddings,
            feed_z_method=["first"]
        )
    elif (hparams.gen.tm.enc.style == 'transformer') ^ (hparams.gen.tm.dec.style == 'transformer'):
        raise NotImplementedError("When using a transformer TM, both encoder and decoder have to be transformer")
    else:
        feed_z_method = ['init_encoder', 'init_decoder']
        if hparams.gen.tm.dec.feed_z:
            feed_z_method.append('cat')
        translation_model = AttentionBasedTM(
            src_embedder=src_embedder,
            tgt_embedder=tgt_embedder,
            tgt_sos_idx=vocab_tgt[SOS_TOKEN],
            tgt_eos_idx=vocab_tgt[EOS_TOKEN],
            encoder=encoder,
            decoder=decoder,
            latent_size=hparams.prior.latent_size,
            dropout=hparams.dropout,
            feed_z_method=feed_z_method,
            tied_embeddings=hparams.gen.tm.dec.tied_embeddings
        )

    return translation_model


def create_aux_translation_models(src_embedder, tgt_embedder, hparams) -> Dict[str, GenerativeTM]:
    tms = dict()
    if hparams.aux.bow_tl:
        tms['bow'] = IndependentTM(
            latent_size=hparams.prior.latent_size,
            embedder=tgt_embedder,
            tied_embeddings=hparams.gen.tm.dec.tied_embeddings)
    if hparams.aux.MADE_tl:
        tms['made'] = CorrelatedBernoullisTM(
            vocab_size=tgt_embedder.num_embeddings,
            latent_size=hparams.prior.latent_size,
            hidden_sizes=[hparams.aux.hidden_size, hparams.aux.hidden_size],
            pad_idx=tgt_embedder.padding_idx,
            num_masks=1,
            resample_mask_every=0)
    if hparams.aux.count_MADE_tl:
        tms['count_made'] = CorrelatedPoissonsTM(
            vocab_size=tgt_embedder.num_embeddings,
            latent_size=hparams.prior.latent_size,
            hidden_sizes=[hparams.aux.hidden_size, hparams.aux.hidden_size],
            pad_idx=tgt_embedder.padding_idx,
            num_masks=1,
            resample_mask_every=0)
    if hparams.aux.shuffle_lm_tl:
        raise NotImplementedError("This is not yet supported")
    if hparams.aux.ibm1:
        tms['ibm1'] = IBM1TM(
            src_embed=src_embedder,
            latent_size=hparams.prior.latent_size,
            hidden_size=hparams.aux.hidden_size,
            src_vocab_size=src_embedder.num_embeddings,
            tgt_vocab_size=tgt_embedder.num_embeddings,
            pad_idx=tgt_embedder.padding_idx)
    return tms


def create_inference_model(src_embedder, tgt_embedder, latent_sizes, hparams) -> InferenceModel:
    """Create an inference model and configure its encoder"""
    if not hparams.inf.inf3:
        # Inference components
        inf_encoder = get_inference_encoder(
            encoder_style=hparams.inf.style,
            conditioning_context=hparams.inf.conditioning,
            embedder_x=src_embedder,
            embedder_y=tgt_embedder,
            hidden_size=hparams.inf.rnn.hidden_size,
            rnn_bidirectional=hparams.inf.rnn.bidirectional,
            rnn_num_layers=hparams.inf.rnn.num_layers,
            rnn_cell_type=hparams.inf.rnn.cell_type,
            transformer_heads=hparams.inf.transformer.num_heads,
            transformer_layers=hparams.inf.transformer.num_layers,
            transformer_hidden=hparams.inf.transformer.hidden_size,
            nli_shared_size=hparams.emb.size,
            nli_max_distance=20, # TODO: generalise
            dropout=hparams.dropout,
            composition=hparams.inf.composition)
        if len(latent_sizes) != len(hparams.posterior.family.split(";")):
            raise ValueError("You need as many posteriors as you have priors")
        conditioners = []
        for family, latent_size in zip(hparams.posterior.family.split(";"), latent_sizes):
            family = family.strip().lower()
            if family == "gaussian":
                conditioners.append(NormalLayer(inf_encoder.output_size, hparams.inf.rnn.hidden_size, latent_size))
            elif family == "kumaraswamy":
                conditioners.append(KumaraswamyLayer(inf_encoder.output_size, hparams.inf.rnn.hidden_size, latent_size))
            elif family == "hardkumaraswamy":
                conditioners.append(HardKumaraswamyLayer(inf_encoder.output_size, hparams.inf.rnn.hidden_size, latent_size))
            else:
                raise NotImplementedError("I cannot design %s posterior approximation." % family)
        inf_model = BasicInferenceModel(
            latent_size=hparams.prior.latent_size,
            conditioner=conditioners[0] if len(conditioners) == 1 else ProductOfConditionalsLayer(conditioners),
            encoder=inf_encoder)
    else:  # create 3 inference models and wrap them around a single container
        # TODO: compatible with multiple priors?
        enc_styles = hparams.inf.inf3.split(',')
        if len(enc_styles) != 3:
            raise ValueError("Specify exactly 3 comma-separated encoder styles, got '%s'" % hparams.inf.inf3)
        encoder_x = get_inference_encoder(
            encoder_style=enc_styles[0],
            conditioning_context='x',
            embedder_x=src_embedder,
            embedder_y=tgt_embedder,
            hidden_size=hparams.inf.rnn.hidden_size,
            rnn_bidirectional=hparams.inf.rnn.bidirectional,
            rnn_num_layers=hparams.inf.rnn.num_layers,
            rnn_cell_type=hparams.inf.rnn.cell_type,
            transformer_heads=hparams.inf.transformer.num_heads,
            transformer_layers=hparams.inf.transformer.num_layers,
            transformer_hidden=hparams.inf.transformer.hidden_size,
            nli_shared_size=hparams.emb.size,
            nli_max_distance=20,  # TODO: generalise
            dropout=hparams.dropout,
            composition=hparams.inf.composition)
        encoder_y = get_inference_encoder(
            encoder_style=enc_styles[1],
            conditioning_context='y',
            embedder_x=src_embedder,
            embedder_y=tgt_embedder,
            hidden_size=hparams.inf.rnn.hidden_size,
            rnn_bidirectional=hparams.inf.rnn.bidirectional,
            rnn_num_layers=hparams.inf.rnn.num_layers,
            rnn_cell_type=hparams.inf.rnn.cell_type,
            transformer_heads=hparams.inf.transformer.num_heads,
            transformer_layers=hparams.inf.transformer.num_layers,
            transformer_hidden=hparams.inf.transformer.hidden_size,
            nli_shared_size=hparams.emb.size,
            nli_max_distance=20,  # TODO: generalise
            dropout=hparams.dropout,
            composition=hparams.inf.composition)
        if enc_styles[2] == 'comb':
            encoder_xy = combine_inference_encoders(encoder_x, encoder_y, hparams.inf.inf3_comb_composition)
        else:
            encoder_xy = get_inference_encoder(
                encoder_style=enc_styles[2],
                conditioning_context='xy',
                embedder_x=src_embedder,
                embedder_y=tgt_embedder,
                hidden_size=hparams.inf.rnn.hidden_size,
                rnn_bidirectional=hparams.inf.rnn.bidirectional,
                rnn_num_layers=hparams.inf.rnn.num_layers,
                rnn_cell_type=hparams.inf.rnn.cell_type,
                transformer_heads=hparams.inf.transformer.num_heads,
                transformer_layers=hparams.inf.transformer.num_layers,
                transformer_hidden=hparams.inf.transformer.hidden_size,
                nli_shared_size=hparams.emb.size,
                nli_max_distance=20,  # TODO: generalise
                dropout=hparams.dropout,
                composition=hparams.inf.composition)
        inf_model = SwitchingInferenceModel(
            BasicInferenceModel(
                family=hparams.posterior.family,
                latent_size=hparams.prior.latent_size,
                hidden_size=hparams.inf.rnn.hidden_size,
                encoder=encoder_x),
            BasicInferenceModel(
                family=hparams.posterior.family,
                latent_size=hparams.prior.latent_size,
                hidden_size=hparams.inf.rnn.hidden_size,
                encoder=encoder_y),
            BasicInferenceModel(
                family=hparams.posterior.family,
                latent_size=hparams.prior.latent_size,
                hidden_size=hparams.inf.rnn.hidden_size,
                encoder=encoder_xy),
            )
    return inf_model


def create_constraints(hparams):
    constraints = dict()

    if hparams.loss.type == "ELBO":
        if hparams.loss.ELBO.mdr > 0:
            constraints['MDR'] = Constraint(hparams.loss.ELBO.mdr, 'ge', name='MDR')

    elif hparams.loss.type == "InfoVAE":
        # Add optional InfoVAE constraints here.
        pass

    elif hparams.loss.type == "LagVAE":
        constraints['ELBO'] = Constraint(hparams.loss.LagVAE.max_elbo, 'le', name='ELBO')
        constraints['MMD'] = Constraint(hparams.loss.LagVAE.max_mmd, 'le', name='MMD')

    elif hparams.loss.type == "IWAE":
        pass

    else:
        raise ValueError(f"Unknown loss type: {hparams.loss.type}")

    return constraints


def create_senvae_model(hparams, vocab_src):
    # Generative components
    src_embedder = torch.nn.Embedding(vocab_src.size(), hparams.emb.size, padding_idx=vocab_src[PAD_TOKEN])
    language_model = create_language_model(vocab_src, src_embedder, hparams)

    # Auxiliary generative components
    aux_lms = create_aux_language_models(vocab_src, src_embedder, hparams)
    if aux_lms:
        raise NotImplementedError("Aux losses are not yet supported with the new loss functions. See Issue #17.")

    priors = []
    n_priors = len(hparams.prior.family.split(";"))
    if hparams.prior.latent_sizes:
        latent_sizes = [int(size) for size in re.split('[ ;:,]+', hparams.prior.latent_sizes.strip())]
        if len(latent_sizes) != n_priors:
            raise ValueError("You need to specify a latent_size for each prior using --latent_sizes 'list'")
        if sum(latent_sizes) != hparams.prior.latent_size:
            raise ValueError("The sum of latent_sizes must equal latent_size")
    else:
        if hparams.prior.latent_size % n_priors != 0:
            raise ValueError("Use a latent size multiple of the number of priors")
        latent_sizes = [hparams.prior.latent_size // n_priors] * n_priors
    for prior_family, prior_params, latent_size in zip(hparams.prior.family.split(";"), hparams.prior.params.split(";"),
                                                       latent_sizes):
        prior_params = [float(param) for param in prior_params.split()]
        priors.append(create_prior(prior_family, latent_size, prior_params))

    inf_model = create_inference_model(
        DetachedEmbeddingLayer(src_embedder) if hparams.emb.shared else torch.nn.Embedding(
            src_embedder.num_embeddings, src_embedder.embedding_dim, padding_idx=src_embedder.padding_idx), None,
        latent_sizes,
        hparams)

    constraints = create_constraints(hparams)

    model = SenVAE(
        latent_size=hparams.prior.latent_size,
        src_embedder=src_embedder,
        language_model=language_model,
        inference_model=inf_model,
        dropout=hparams.dropout,
        feed_z=None,
        tied_embeddings=None,
        prior=priors[0] if len(priors) == 1 else ProductOfPriorsLayer(priors),
        constraints=constraints,
        aux_lms=aux_lms,
        mixture_likelihood=hparams.likelihood.mixture,
        mixture_likelihood_dir_prior=hparams.likelihood.mixture_dir_prior)
    return model

def create_model(hparams, vocab_src, vocab_tgt):
    # Generative components
    src_embedder = torch.nn.Embedding(vocab_src.size(), hparams.emb.size, padding_idx=vocab_src[PAD_TOKEN])
    tgt_embedder = torch.nn.Embedding(vocab_tgt.size(), hparams.emb.size, padding_idx=vocab_tgt[PAD_TOKEN])

    language_model = create_language_model(vocab_src, src_embedder, hparams)

    # Auxiliary generative components
    aux_lms = create_aux_language_models(vocab_src, src_embedder, hparams)
    aux_tms = create_aux_translation_models(src_embedder, tgt_embedder, hparams)
    if aux_lms or aux_tms:
        raise NotImplementedError("Aux losses are not yet supported with the new loss functions. See Issue #17.")

    translation_model = create_translation_model(vocab_tgt, src_embedder, tgt_embedder, hparams)

    priors = []
    n_priors = len(hparams.prior.family.split(";"))
    if hparams.prior.latent_sizes:
        latent_sizes = [int(size) for size in re.split('[ ;:,]+', hparams.prior.latent_sizes.strip())]
        if len(latent_sizes) != n_priors:
            raise ValueError("You need to specify a latent_size for each prior using --latent_sizes 'list'")
        if sum(latent_sizes) != hparams.prior.latent_size:
            raise ValueError("The sum of latent_sizes must equal latent_size")
    else:
        if hparams.prior.latent_size % n_priors != 0:
            raise ValueError("Use a latent size multiple of the number of priors")
        latent_sizes = [hparams.prior.latent_size // n_priors] * n_priors
    for prior_family, prior_params, latent_size in zip(hparams.prior.family.split(";"), hparams.prior.params.split(";"), latent_sizes):
        prior_params = [float(param) for param  in prior_params.split()]
        priors.append(create_prior(prior_family, latent_size, prior_params))

    inf_model = create_inference_model(
        DetachedEmbeddingLayer(src_embedder) if hparams.emb.shared else torch.nn.Embedding(
            src_embedder.num_embeddings, src_embedder.embedding_dim, padding_idx=src_embedder.padding_idx),
        DetachedEmbeddingLayer(tgt_embedder) if hparams.emb.shared else torch.nn.Embedding(
            tgt_embedder.num_embeddings, tgt_embedder.embedding_dim, padding_idx=tgt_embedder.padding_idx),
        latent_sizes,
        hparams)

    constraints = create_constraints(hparams)

    model = AEVNMT(
        latent_size=hparams.prior.latent_size,
        src_embedder=src_embedder,
        tgt_embedder=tgt_embedder,
        language_model=language_model,
        translation_model=translation_model,
        inference_model=inf_model,
        dropout=hparams.dropout,
        feed_z=None,
        tied_embeddings=None,
        prior=priors[0] if len(priors) == 1 else ProductOfPriorsLayer(priors),
        constraints=constraints,
        aux_lms=aux_lms,
        aux_tms=aux_tms,
        mixture_likelihood=hparams.likelihood.mixture,
        mixture_likelihood_dir_prior=hparams.likelihood.mixture_dir_prior)
    return model


def create_loss(hparams):
    if hparams.loss.type == "ELBO":
        loss = loss_functions.ELBOLoss(
            kl_weight=hparams.loss.ELBO.beta,
            kl_annealing_steps=hparams.loss.ELBO.kl_annealing_steps,
            free_nats=hparams.loss.ELBO.free_nats,
            label_smoothing_x=hparams.gen.lm.label_smoothing,
            label_smoothing_y=hparams.gen.tm.label_smoothing)
    elif hparams.loss.type == "InfoVAE":
        loss = loss_functions.InfoVAELoss(
            info_alpha=hparams.loss.InfoVAE.alpha,
            info_lambda=hparams.loss.InfoVAE.lamb,
            label_smoothing_x=hparams.gen.lm.label_smoothing,
            label_smoothing_y=hparams.gen.tm.label_smoothing
        )
    elif hparams.loss.type == "LagVAE":
        loss = loss_functions.LagVAELoss(
            alpha=hparams.loss.LagVAE.alpha,
            label_smoothing_x=hparams.gen.lm.label_smoothing,
            label_smoothing_y=hparams.gen.tm.label_smoothing
        )
    elif hparams.loss.type == "IWAE":
        loss = loss_functions.IWAELoss(
            num_samples=hparams.loss.IWAE.num_samples,
            label_smoothing_x=hparams.gen.lm.label_smoothing,
            label_smoothing_y=hparams.gen.tm.label_smoothing
        )
    else:
        raise ValueError(f"Invalid loss type: {hparams.loss.type}")
    return loss

def train_step(model, x_in, x_out, seq_mask_x, seq_len_x, noisy_x_in, y_in, y_out,
               seq_mask_y, seq_len_y, noisy_y_in, hparams, step, summary_writer=None):


    # Use q(z|x) for training to sample a z.
    qz = model.approximate_posterior(x_in, seq_mask_x, seq_len_x, y_in, seq_mask_y, seq_len_y)
    z = qz.rsample()

    # Compute the translation and language model logits.
    tm_likelihood, lm_likelihood, _, aux_lm_likelihoods, aux_tm_likelihoods = model(noisy_x_in, seq_mask_x, seq_len_x, noisy_y_in, z)

    # Do linear annealing of the KL over KL_annealing_steps if set.
    KL_weight = hparams.kl.weight
    if hparams.kl.annealing_steps > 0:
        KL_weight = min(KL_weight, (KL_weight / hparams.kl.annealing_steps) * step)

    # Compute the loss.
    loss_cfg = None
    #if step < 20000:
    #    loss_cfg = {'lm/bow', }  #, 'tm/bow'}
    #else:  #if step < 20000:
    #    loss_cfg = {'lm/made', } #, 'tm/made'}
    #elif step < 3000:
    #    loss_cfg = {'lm/made', 'lm/made_count', 'tm/made', 'tm/made_count'}
    #elif step < 4500:
    #    loss_cfg = {'lm/made', 'lm/made_count', 'lm/main', 'tm/made', 'tm/made_count', 'tm/main'}
    #else:
    #    loss_cfg = {'lm/main', 'tm/main'}
    loss = model.loss(tm_likelihood, lm_likelihood, y_out, x_out, qz,
                      free_nats=hparams.kl.free_nats,
                      KL_weight=KL_weight,
                      mmd_weight=hparams.loss.mmd_weight,
                      reduction="mean",
                      smoothing_x=hparams.gen.lm.label_smoothing,
                      smoothing_y=hparams.gen.tm.label_smoothing,
                      aux_lm_likelihoods=aux_lm_likelihoods,
                      aux_tm_likelihoods=aux_tm_likelihoods,
                      loss_cfg=loss_cfg)

    if summary_writer and step % hparams.print_every == 0:
        summary_writer.add_histogram("posterior/z", z, step)
        for param_name, param_value in get_named_params(qz):
            summary_writer.add_histogram("posterior/%s" % param_name, param_value, step)
        pz = model.prior()
        # This part is perhaps not necessary for a simple prior (e.g. Gaussian),
        #  but it's useful for more complex priors (e.g. mixtures and NFs)
        prior_sample = pz.sample(torch.Size([z.size(0)]))
        summary_writer.add_histogram("prior/z", prior_sample, step)
        for param_name, param_value in get_named_params(pz):
            summary_writer.add_histogram("prior/%s" % param_name, param_value, step)

    return loss

def validate(model, val_data, vocab_src, vocab_tgt, device, hparams, step, title='xy', summary_writer=None):
    if isinstance(model, ParallelWrapper):
        model = model.module.model
    model.eval()

    # Create the validation dataloader. We can just bucket.
    val_dl = DataLoader(val_data, batch_size=hparams.batch_size,
                        shuffle=False, num_workers=4)
    val_dl = BucketingParallelDataLoader(val_dl)

    val_ppl, val_KL, val_NLLs = _evaluate_perplexity(model, val_dl, vocab_src, vocab_tgt, device)
    val_NLL = val_NLLs['joint/main']
    val_bleu, inputs, refs, hyps = _evaluate_bleu(model, val_dl, vocab_src, vocab_tgt,
                                                  device, hparams)

    random_idx = np.random.choice(len(inputs))
    #nll_str = ' '.join('-- validation NLL {} = {:.2f}'.format(comp_name, comp_value)  for comp_name, comp_value in sorted(val_NLLs.items()))
    nll_str = f""
    # - log P(x|z) for the various source LM decoders
    for comp_name, comp_nll in sorted(val_NLLs.items()):
        if comp_name.startswith('lm/'):
            nll_str += f" -- {comp_name} = {comp_nll:,.2f}"
    # - log P(y|z,x) for the various translation decoders
    for comp_name, comp_nll in sorted(val_NLLs.items()):
        if comp_name.startswith('tm/'):
            nll_str += f" -- {comp_name} = {comp_nll:,.2f}"

    kl_str = f"-- KL = {val_KL.sum():.2f}"
    if isinstance(model.prior(), ProductOfDistributions):
        for i, p in enumerate(model.prior().distributions):
            kl_str += f" -- KL{i} = {val_KL[i]:.2f}"

    print(f"direction = {title}\n"
          f"validation perplexity = {val_ppl:,.2f}"
          f" -- BLEU = {val_bleu:.2f}"
          f" {kl_str}"
          f" {nll_str}\n"
          f"- Source: {inputs[random_idx]}\n"
          f"- Target: {refs[random_idx]}\n"
          f"- Prediction: {hyps[random_idx]}")

    if hparams.draw_translations > 0:
        random_idx = np.random.choice(len(inputs))
        dl = DataLoader([val_data[random_idx] for _ in range(hparams.draw_translations)], batch_size=hparams.batch_size, shuffle=False, num_workers=4)
        dl = BucketingParallelDataLoader(dl)
        i, r, hs = _draw_translations(model, dl, vocab_src, vocab_tgt, device, hparams)
        print("Posterior samples")
        print(f"- Input: {i[0]}")
        print(f"- Reference: {r[0]}")
        for h in hs:
            print(f"- Translation: {h}")

    # Write validation summaries.
    if summary_writer is not None:
        summary_writer.add_scalar(f"{title}/validation/BLEU", val_bleu, step)
        summary_writer.add_scalar(f"{title}/validation/perplexity", val_ppl, step)
        summary_writer.add_scalar(f"{title}/validation/KL", val_KL.sum(), step)
        if isinstance(model.prior(), ProductOfDistributions):
            for i, _ in enumerate(model.prior().distributions):
                summary_writer.add_scalar(f"{title}/validation/KL{i}", val_KL[i], step)
        for comp_name, comp_value in val_NLLs.items():
            summary_writer.add_scalar(f"{title}/validation/NLL/{comp_name}", comp_value, step)

        # Log the attention weights of the first validation sentence.
        with torch.no_grad():
            val_sentence_x, val_sentence_y = val_data[0]
            x_in, _, seq_mask_x, seq_len_x = create_batch([val_sentence_x], vocab_src, device)
            y_in, y_out, seq_mask_y, seq_len_y = create_batch([val_sentence_y], vocab_tgt, device)
            z = model.approximate_posterior(x_in, seq_mask_x, seq_len_x, y_in, seq_mask_y, seq_len_y).sample()
            _, _, state, _, _ = model(x_in, seq_mask_x, seq_len_x, y_in, z)
            if 'att_weights' in state:
                att_weights = state['att_weights'].squeeze().cpu().numpy()
            else:
                att_weights = None
        src_labels = batch_to_sentences(x_in, vocab_src, no_filter=True)[0].split()
        tgt_labels = batch_to_sentences(y_out, vocab_tgt, no_filter=True)[0].split()
        if att_weights is not None:
            # TODO add attention summary for Transformer
            attention_summary(src_labels, tgt_labels, att_weights, summary_writer,
                            f"{title}/validation/attention", step)

    return {'bleu': val_bleu, 'likelihood': -val_NLL, 'nll': val_NLL, 'ppl': val_ppl}


def validate_senvae(model, val_data, vocab_src, device, hparams, step, title='x', summary_writer=None):
    if isinstance(model, ParallelWrapper):
        model = model.module.model
    model.eval()

    # Create the validation dataloader. We can just bucket.
    val_dl = DataLoader(val_data, batch_size=hparams.batch_size,
                        shuffle=False, num_workers=4)
    val_dl = BucketingTextDataLoader(val_dl)

    val_ppl, val_KL, val_NLLs = _evaluate_perplexity(model, val_dl, vocab_src, None, device)
    val_NLL = val_NLLs['joint/main']

    # nll_str = ' '.join('-- validation NLL {} = {:.2f}'.format(comp_name, comp_value)  for comp_name, comp_value in sorted(val_NLLs.items()))
    nll_str = f""
    # - log P(x|z) for the various source LM decoders
    for comp_name, comp_nll in sorted(val_NLLs.items()):
        if comp_name.startswith('lm/'):
            nll_str += f" -- {comp_name} = {comp_nll:,.2f}"
    # - log P(y|z,x) for the various translation decoders
    for comp_name, comp_nll in sorted(val_NLLs.items()):
        if comp_name.startswith('tm/'):
            nll_str += f" -- {comp_name} = {comp_nll:,.2f}"

    kl_str = f"-- KL = {val_KL.sum():.2f}"
    if isinstance(model.prior(), ProductOfDistributions):
        for i, p in enumerate(model.prior().distributions):
            kl_str += f" -- KL{i} = {val_KL[i]:.2f}"

    print(f"direction = {title}\n"
          f"validation perplexity = {val_ppl:,.2f}"
          f" {kl_str}"
          f" {nll_str}\n")


    # Write validation summaries.
    if summary_writer is not None:
        summary_writer.add_scalar(f"{title}/validation/perplexity", val_ppl, step)
        summary_writer.add_scalar(f"{title}/validation/KL", val_KL.sum(), step)
        if isinstance(model.prior(), ProductOfDistributions):
            for i, _ in enumerate(model.prior().distributions):
                summary_writer.add_scalar(f"{title}/validation/KL{i}", val_KL[i], step)
        for comp_name, comp_value in val_NLLs.items():
            summary_writer.add_scalar(f"{title}/validation/NLL/{comp_name}", comp_value, step)


    return {'bleu': 0.0, 'likelihood': -val_NLL, 'nll': val_NLL, 'ppl': val_ppl}


def translate(model, input_sentences, vocab_src, vocab_tgt, device, hparams, deterministic=True):
    # TODO: this code should be in the translation model class
    model.eval()
    with torch.no_grad():
        x_in, _, seq_mask_x, seq_len_x = create_batch(input_sentences, vocab_src, device)

        # For translation we use the approximate posterior mean.
        qz = model.approximate_posterior(x_in, seq_mask_x, seq_len_x,
                y=x_in, seq_mask_y=seq_mask_x, seq_len_y=seq_len_x) # TODO: here we need a prediction net!
        # TODO: restore some form of deterministic decoding
        #z = qz.mean if deterministic else qz.sample()
        z = qz.sample()

        if isinstance(model.translation_model, TransformerTM):
            encoder_outputs, seq_len_x = model.translation_model.encode(x_in, seq_len_x, z)
            encoder_final = None
            hidden = None
        else:
            encoder_outputs, encoder_final = model.translation_model.encode(x_in, seq_len_x, z)
            hidden = model.translation_model.init_decoder(encoder_outputs, encoder_final, z)

        if hparams.decoding.sample:
            raw_hypothesis = model.translation_model.sample(x_in, seq_mask_x, seq_len_x, z,
               max_len=hparams.decoding.max_length, greedy=False)

        elif hparams.decoding.beam_width <= 1:
            raw_hypothesis = model.translation_model.sample(x_in, seq_mask_x, seq_len_x, z,
               max_len=hparams.decoding.max_length, greedy=True)

        else:
            raw_hypothesis = beam_search(
                model.translation_model.decoder,
                model.translation_model.tgt_embed,
                model.translation_model.generate,
                vocab_tgt.size(), hidden, encoder_outputs,
                encoder_final, seq_mask_x, seq_len_x,
                vocab_tgt[SOS_TOKEN], vocab_tgt[EOS_TOKEN],
                vocab_tgt[PAD_TOKEN], hparams.decoding.beam_width,
                hparams.decoding.length_penalty_factor,
                hparams.decoding.max_length,
                z)

    hypothesis = batch_to_sentences(raw_hypothesis, vocab_tgt)
    return hypothesis

#TODO: change me
def generate_senvae(model, num_samples, vocab_src, device, hparams, deterministic=True):

    model.eval()
    with torch.no_grad():
        qz=model.prior().expand((num_samples,))

        # TODO: restore some form of deterministic decoding
        #z = qz.mean if deterministic else qz.sample()
        z = qz.sample()

        if isinstance(model.translation_model, TransformerTM):
            encoder_outputs, seq_len_x = model.translation_model.encode(x_in, seq_len_x, z)
            encoder_final = None
            hidden = None
        else:
            encoder_outputs, encoder_final = model.translation_model.encode(x_in, seq_len_x, z)
            hidden = model.translation_model.init_decoder(encoder_outputs, encoder_final, z)

        if hparams.decoding.sample:
            raw_hypothesis = model.translation_model.sample(x_in, seq_mask_x, seq_len_x, z,
               max_len=hparams.decoding.max_length, greedy=False)

        elif hparams.decoding.beam_width <= 1:
            raw_hypothesis = model.translation_model.sample(x_in, seq_mask_x, seq_len_x, z,
               max_len=hparams.decoding.max_length, greedy=True)

        else:
            raw_hypothesis = beam_search(
                model.translation_model.decoder,
                model.translation_model.tgt_embed,
                model.translation_model.generate,
                vocab_tgt.size(), hidden, encoder_outputs,
                encoder_final, seq_mask_x, seq_len_x,
                vocab_tgt[SOS_TOKEN], vocab_tgt[EOS_TOKEN],
                vocab_tgt[PAD_TOKEN], hparams.decoding.beam_width,
                hparams.decoding.length_penalty_factor,
                hparams.decoding.max_length,
                z)

    hypothesis = batch_to_sentences(raw_hypothesis, vocab_src)
    return hypothesis


def _evaluate_bleu(model, val_dl, vocab_src, vocab_tgt, device, hparams):
    model.eval()
    with torch.no_grad():
        inputs = []
        references = []
        model_hypotheses = []
        for sentences_x, sentences_y in val_dl:
            hypothesis = translate(model, sentences_x, vocab_src, vocab_tgt, device, hparams)

            # Keep track of inputs, references and model hypotheses.
            inputs += sentences_x.tolist()
            references += sentences_y.tolist()
            model_hypotheses += hypothesis.tolist()

    bleu = compute_bleu(model_hypotheses, references, subword_token=hparams.subword_token)
    return bleu, inputs, references, model_hypotheses

def _evaluate_perplexity(model, val_dl, vocab_src, vocab_tgt, device):
    model.eval()
    with torch.no_grad():
        num_predictions = 0
        num_sentences = 0
        log_marginal = defaultdict(float)
        total_KL = 0.
        n_samples = 10
        for tup in val_dl:
            if vocab_tgt is not None:
                sentences_x, sentences_y = tup
            else:
                sentences_x=tup
            x_in, x_out, seq_mask_x, seq_len_x = create_batch(sentences_x, vocab_src, device)
            if vocab_tgt is not None:
                y_in, y_out, seq_mask_y, seq_len_y = create_batch(sentences_y, vocab_tgt, device)
            else:
                y_in, y_out, seq_mask_y, seq_len_y = None,None,None,None

            # Infer q(z|x) for this batch.
            qz = model.approximate_posterior(x_in, seq_mask_x, seq_len_x, y_in, seq_mask_y, seq_len_y)
            pz = model.prior()
            if isinstance(qz, ProductOfDistributions):
                total_KL += torch.cat(
                    [kl_divergence(qi, pi).sum(0).unsqueeze(-1) for qi, pi in zip(qz.distributions, pz.distributions)],
                    -1)
            else:
                total_KL += kl_divergence(qz, pz).sum(0)

            # Take s importance samples from q(z|x):
            # log int{p(x, y, z) dz} ~= log sum_z{p(x, y, z) / q(z|x)} where z ~ q(z|x)
            batch_size = x_in.size(0)
            batch_log_marginals = defaultdict(lambda: torch.zeros(n_samples, batch_size))

            for s in range(n_samples):

                # z ~ q(z|x)
                z = qz.sample()

                # Compute the logits according to this sample of z.
                if isinstance(model,SenVAE):
                    tm_likelihood=None
                    aux_tm_likelihoods=None
                    lm_likelihood, _, aux_lm_likelihoods  = model(x_in, seq_mask_x, seq_len_x, z)
                else:
                    tm_likelihood, lm_likelihood, _, aux_lm_likelihoods, aux_tm_likelihoods = model(x_in, seq_mask_x, seq_len_x, y_in, z)

                # Compute log P(y|x, z_s)
                if tm_likelihood is not None:
                    log_tm_prob = model.translation_model.log_prob(tm_likelihood, y_out)
                else:
                    log_tm_prob =None

                # Compute log P(x|z_s)
                log_lm_prob = model.language_model.log_prob(lm_likelihood, x_out)

                # Compute prior probability log P(z_s) and importance weight q(z_s|x)
                log_pz = pz.log_prob(z)
                log_qz = qz.log_prob(z)

                # Estimate the importance weighted estimate of (the log of) P(x, y)
                batch_log_marginals['joint/main'][s] =  log_lm_prob + log_pz - log_qz
                batch_log_marginals['lm/main'][s] = log_lm_prob + log_pz - log_qz
                if log_tm_prob is not None:
                    batch_log_marginals['tm/main'][s] = log_tm_prob + log_pz - log_qz
                    batch_log_marginals['joint/main'][s]+=log_tm_prob

                for aux_comp, aux_px_z in aux_lm_likelihoods.items():
                    batch_log_marginals['lm/' + aux_comp][s] = model.log_likelihood_lm(aux_comp, aux_px_z, x_out) + log_pz - log_qz
                if aux_tm_likelihoods is not None:
                    for aux_comp, aux_py_xz in aux_tm_likelihoods.items():
                        batch_log_marginals['tm/' + aux_comp][s] = model.log_likelihood_tm(aux_comp, aux_py_xz, y_out) + log_pz - log_qz

            for comp_name, log_marginals in batch_log_marginals.items():
                # Average over all samples.
                batch_avg = torch.logsumexp(log_marginals, dim=0) - torch.log(torch.Tensor([n_samples]))
                log_marginal[comp_name] = log_marginal[comp_name] + batch_avg.sum().item()

            num_sentences += batch_size
            num_predictions += (seq_len_x.sum() + (seq_len_y.sum() if vocab_tgt is not None else 0.0)).item()

    val_NLL = -log_marginal['joint/main']
    val_perplexity = np.exp(val_NLL / num_predictions)

    NLLs = {comp_name: -value / num_sentences for comp_name, value in log_marginal.items()}

    return val_perplexity, total_KL/num_sentences, NLLs


def product_of_gaussians(fwd_base: Normal, bwd_base: Normal) -> Normal:
    u1, s1, var1 = fwd_base.mean, fwd_base.stddev, fwd_base.variance
    u2, s2, var2 = bwd_base.mean, bwd_base.stddev, bwd_base.variance
    u = (var2 * u1 + var1 * u2) / (var1 + var2)
    s = 1. / (1. / var1 + 1. / var2)
    return Normal(u, s)


def mixture_of_gaussians(fwd_base: Normal, bwd_base: Normal) -> Normal:
    # [batch_size]
    selectors = torch.rand(fwd_base.mean.size(0), device=fwd_base.mean.device) >= 0.5
    # [batch_size, latent_size]
    selectors = selectors.unsqueeze(-1).repeat([1, fwd_base.mean.size(1)])
    u = torch.where(selectors, fwd_base.mean, bwd_base.mean)
    s = torch.where(selectors, fwd_base.stddev, bwd_base.stddev)
    return Normal(u, s)
