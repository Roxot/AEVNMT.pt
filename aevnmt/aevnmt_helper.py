import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal

from aevnmt.data import BucketingParallelDataLoader, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN
from aevnmt.data import create_batch, batch_to_sentences
from aevnmt.components import RNNEncoder, beam_search, greedy_decode, sampling_decode
from aevnmt.models import AEVNMT, RNNLM
from .train_utils import create_attention, create_decoder, attention_summary, compute_bleu

from torch.utils.data import DataLoader


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


def create_model(hparams, vocab_src, vocab_tgt):
    rnnlm = RNNLM(vocab_size=vocab_src.size(),
                  emb_size=hparams.emb_size,
                  hidden_size=hparams.hidden_size,
                  pad_idx=vocab_src[PAD_TOKEN],
                  dropout=hparams.dropout,
                  num_layers=hparams.num_dec_layers,
                  cell_type=hparams.cell_type,
                  tied_embeddings=hparams.tied_embeddings)
    encoder = create_encoder(hparams)
    attention = create_attention(hparams)
    decoder = create_decoder(attention, hparams)
    model = AEVNMT(tgt_vocab_size=vocab_tgt.size(),
                   emb_size=hparams.emb_size,
                   latent_size=hparams.latent_size,
                   encoder=encoder,
                   decoder=decoder,
                   language_model=rnnlm,
                   pad_idx=vocab_tgt[PAD_TOKEN],
                   dropout=hparams.dropout,
                   tied_embeddings=hparams.tied_embeddings)
    return model

def train_step(model, x_in, x_out, seq_mask_x, seq_len_x, noisy_x_in, y_in, y_out, seq_mask_y, seq_len_y, noisy_y_in,
               hparams, step):

    # Use q(z|x) for training to sample a z.
    qz = model.approximate_posterior(x_in, seq_mask_x, seq_len_x)
    z = qz.rsample()

    # Compute the translation and language model logits.
    tm_logits, lm_logits, _ = model(noisy_x_in, seq_mask_x, seq_len_x, noisy_y_in, z)

    # Do linear annealing of the KL over KL_annealing_steps if set.
    if hparams.KL_annealing_steps > 0:
        KL_weight = min(1., (1.0 / hparams.KL_annealing_steps) * step)
    else:
        KL_weight = 1.

    # Compute the loss.
    loss = model.loss(tm_logits, lm_logits, y_out, x_out, qz,
                      free_nats=hparams.KL_free_nats,
                      KL_weight=KL_weight,
                      reduction="mean")
    return loss

def validate(model, val_data, vocab_src, vocab_tgt, device, hparams, step, title='xy', summary_writer=None):
    model.eval()

    # Create the validation dataloader. We can just bucket.
    val_dl = DataLoader(val_data, batch_size=hparams.batch_size,
                        shuffle=False, num_workers=4)
    val_dl = BucketingParallelDataLoader(val_dl)

    val_ppl, val_NLL, val_KL = _evaluate_perplexity(model, val_dl, vocab_src, vocab_tgt, device)
    val_bleu, inputs, refs, hyps = _evaluate_bleu(model, val_dl, vocab_src, vocab_tgt,
                                                  device, hparams)

    random_idx = np.random.choice(len(inputs))
    print(f"direction = {title}\n"
          f"validation perplexity = {val_ppl:,.2f}"
          f" -- validation NLL = {val_NLL:,.2f}"
          f" -- validation BLEU = {val_bleu:.2f}"
          f" -- validation KL = {val_KL:.2f}\n"
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
        summary_writer.add_scalar(f"{title}/validation/NLL", val_NLL, step)
        summary_writer.add_scalar(f"{title}/validation/BLEU", val_bleu, step)
        summary_writer.add_scalar(f"{title}/validation/perplexity", val_ppl, step)
        summary_writer.add_scalar(f"{title}/validation/KL", val_KL, step)

        # Log the attention weights of the first validation sentence.
        with torch.no_grad():
            val_sentence_x, val_sentence_y = val_data[0]
            x_in, _, seq_mask_x, seq_len_x = create_batch([val_sentence_x], vocab_src, device)
            y_in, y_out, _, _ = create_batch([val_sentence_y], vocab_tgt, device)
            z = model.approximate_posterior(x_in, seq_mask_x, seq_len_x).sample()
            _, _, att_weights = model(x_in, seq_mask_x, seq_len_x, y_in, z)
            att_weights = att_weights.squeeze().cpu().numpy()
        src_labels = batch_to_sentences(x_in, vocab_src, no_filter=True)[0].split()
        tgt_labels = batch_to_sentences(y_out, vocab_tgt, no_filter=True)[0].split()
        attention_summary(src_labels, tgt_labels, att_weights, summary_writer,
                          f"{title}/validation/attention", step)

    return {'bleu': val_bleu, 'likelihood': -val_NLL, 'nll': val_NLL, 'ppl': val_ppl}


def translate(model, input_sentences, vocab_src, vocab_tgt, device, hparams, deterministic=True):
    model.eval()
    with torch.no_grad():
        x_in, _, seq_mask_x, seq_len_x = create_batch(input_sentences, vocab_src, device)

        # For translation we use the approximate posterior mean.
        qz = model.approximate_posterior(x_in, seq_mask_x, seq_len_x)
        z = qz.mean if deterministic else qz.sample()

        encoder_outputs, encoder_final = model.encode(x_in, seq_len_x, z)
        hidden = model.init_decoder(encoder_outputs, encoder_final, z)

        if hparams.sample_decoding:
            raw_hypothesis = sampling_decode(model.decoder, model.tgt_embed,
                                           model.generate, hidden,
                                           encoder_outputs, encoder_final,
                                           seq_mask_x, vocab_tgt[SOS_TOKEN], vocab_tgt[EOS_TOKEN],
                                           vocab_tgt[PAD_TOKEN], hparams.max_decoding_length)
        elif hparams.beam_width <= 1:
            raw_hypothesis = greedy_decode(model.decoder, model.tgt_embed,
                                           model.generate, hidden,
                                           encoder_outputs, encoder_final,
                                           seq_mask_x, vocab_tgt[SOS_TOKEN], vocab_tgt[EOS_TOKEN],
                                           vocab_tgt[PAD_TOKEN], hparams.max_decoding_length)
        else:
            raw_hypothesis = beam_search(model.decoder, model.tgt_embed, model.generate,
                                         vocab_tgt.size(), hidden, encoder_outputs,
                                         encoder_final, seq_mask_x,
                                         vocab_tgt[SOS_TOKEN], vocab_tgt[EOS_TOKEN],
                                         vocab_tgt[PAD_TOKEN], hparams.beam_width,
                                         hparams.length_penalty_factor,
                                         hparams.max_decoding_length)

    hypothesis = batch_to_sentences(raw_hypothesis, vocab_tgt)
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
        log_marginal = 0.
        total_KL = 0.
        n_samples = 10
        for sentences_x, sentences_y in val_dl:
            x_in, x_out, seq_mask_x, seq_len_x = create_batch(sentences_x, vocab_src, device)
            y_in, y_out, seq_mask_y, seq_len_y = create_batch(sentences_y, vocab_tgt, device)

            # Infer q(z|x) for this batch.
            qz = model.approximate_posterior(x_in, seq_mask_x, seq_len_x)
            pz = model.prior().expand(qz.mean.size())
            total_KL += torch.distributions.kl.kl_divergence(qz, pz).sum().item()

            # Take s importance samples from q(z|x):
            # log int{p(x, y, z) dz} ~= log sum_z{p(x, y, z) / q(z|x)} where z ~ q(z|x)
            batch_size = x_in.size(0)
            batch_log_marginals = torch.zeros(n_samples, batch_size)
            for s in range(n_samples):

                # z ~ q(z|x)
                z = qz.sample()

                # Compute the logits according to this sample of z.
                tm_logits, lm_logits, _ = model(x_in, seq_mask_x, seq_len_x, y_in, z)

                # Compute log P(y|x, z_s)
                log_tm_prob = F.log_softmax(tm_logits, dim=-1)
                log_tm_prob = torch.gather(log_tm_prob, 2, y_out.unsqueeze(-1)).squeeze()
                log_tm_prob = (seq_mask_y.type_as(log_tm_prob) * log_tm_prob).sum(dim=1)

                # Compute log P(x|z_s)
                log_lm_prob = F.log_softmax(lm_logits, dim=-1)
                log_lm_prob = torch.gather(log_lm_prob, 2, x_out.unsqueeze(-1)).squeeze()
                log_lm_prob = (seq_mask_x.type_as(log_lm_prob) * log_lm_prob).sum(dim=1)

                # Compute prior probability log P(z_s) and importance weight q(z_s|x)
                log_pz = pz.log_prob(z).sum(dim=1) # [B, latent_size] -> [B]
                log_qz = qz.log_prob(z).sum(dim=1)

                # Estimate the importance weighted estimate of (the log of) P(x, y)
                batch_log_marginals[s] = log_tm_prob + log_lm_prob + log_pz - log_qz

            # Average over all samples.
            batch_log_marginal = torch.logsumexp(batch_log_marginals, dim=0) - \
                                 torch.log(torch.Tensor([n_samples]))
            log_marginal += batch_log_marginal.sum().item() # [B] -> []
            num_sentences += batch_size
            num_predictions += (seq_len_x.sum() + seq_len_y.sum()).item()

    val_NLL = -log_marginal
    val_perplexity = np.exp(val_NLL / num_predictions)
    return val_perplexity, val_NLL/num_sentences, total_KL/num_sentences


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
