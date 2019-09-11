import torch
import numpy as np

from aevnmt.data import BucketingParallelDataLoader, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN
from aevnmt.data import create_batch, batch_to_sentences
from aevnmt.components import RNNEncoder, beam_search, greedy_decode, sampling_decode, ancestral_sample
from aevnmt.models import ConditionalNMT
from .train_utils import create_attention, create_decoder, attention_summary, compute_bleu

from torch.utils.data import DataLoader

def create_model(hparams, vocab_src, vocab_tgt):
    encoder = RNNEncoder(emb_size=hparams.emb_size,
                         hidden_size=hparams.hidden_size,
                         bidirectional=hparams.bidirectional,
                         dropout=hparams.dropout,
                         num_layers=hparams.num_enc_layers,
                         cell_type=hparams.cell_type)
    attention = create_attention(hparams)
    decoder = create_decoder(attention, hparams)
    model = ConditionalNMT(src_vocab_size=vocab_src.size(),
                           tgt_vocab_size=vocab_tgt.size(),
                           emb_size=hparams.emb_size,
                           encoder=encoder,
                           decoder=decoder,
                           pad_idx=vocab_tgt[PAD_TOKEN],
                           dropout=hparams.dropout,
                           tied_embeddings=hparams.tied_embeddings)
    return model

def train_step(model, x_in, x_out, seq_mask_x, seq_len_x, noisy_x_in, y_in, y_out, seq_mask_y, seq_len_y, noisy_y_in,
               hparams, step):
    logits, _ = model(noisy_x_in, seq_mask_x, seq_len_x, noisy_y_in)
    loss = model.loss(logits, y_out, reduction="mean")
    return loss

def validate(model, val_data, vocab_src, vocab_tgt, device, hparams, step, summary_writer=None):
    model.eval()

    # Create the validation dataloader. We can just bucket.
    val_dl = DataLoader(val_data, batch_size=hparams.batch_size,
                        shuffle=False, num_workers=4)
    val_dl = BucketingParallelDataLoader(val_dl)

    val_ppl, val_NLL = _evaluate_perplexity(model, val_dl, vocab_src, vocab_tgt, device)
    val_bleu, inputs, refs, hyps = _evaluate_bleu(model, val_dl, vocab_src, vocab_tgt,
                                                  device, hparams)

    random_idx = np.random.choice(len(inputs))
    print(f"validation perplexity = {val_ppl:,.2f}"
          f" -- validation NLL = {val_NLL:,.2f}"
          f" -- validation BLEU = {val_bleu:.2f}\n"
          f"- Source: {inputs[random_idx]}\n"
          f"- Target: {refs[random_idx]}\n"
          f"- Prediction: {hyps[random_idx]}")

    # Write validation summaries.
    if summary_writer is not None:
        summary_writer.add_scalar("validation/NLL", val_NLL, step)
        summary_writer.add_scalar("validation/BLEU", val_bleu, step)
        summary_writer.add_scalar("validation/perplexity", val_ppl, step)

        # Log the attention weights of the first validation sentence.
        with torch.no_grad():
            val_sentence_x, val_sentence_y = val_data[0]
            x_in, _, seq_mask_x, seq_len_x = create_batch([val_sentence_x], vocab_src, device)
            y_in, y_out, _, _ = create_batch([val_sentence_y], vocab_tgt, device)
            _, att_weights = model(x_in, seq_mask_x, seq_len_x, y_in)
            att_weights = att_weights.squeeze().cpu().numpy()
        src_labels = batch_to_sentences(x_in, vocab_src, no_filter=True)[0].split()
        tgt_labels = batch_to_sentences(y_out, vocab_tgt, no_filter=True)[0].split()
        attention_summary(tgt_labels, src_labels, att_weights, summary_writer,
                          "validation/attention", step)

    return {'bleu': val_bleu, 'likelihood': -val_NLL,  'nll': val_NLL, 'ppl': val_ppl}

def translate(model, input_sentences, vocab_src, vocab_tgt, device, hparams):
    model.eval()
    with torch.no_grad():
        x_in, _, seq_mask_x, seq_len_x = create_batch(input_sentences, vocab_src, device)
        encoder_outputs, encoder_final = model.encode(x_in, seq_len_x)
        hidden = model.init_decoder(encoder_outputs, encoder_final)
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
        val_NLL = 0.
        for sentences_x, sentences_y in val_dl:
            x_in, _, seq_mask_x, seq_len_x = create_batch(sentences_x, vocab_src, device)
            y_in, y_out, _, seq_len_y = create_batch(sentences_y, vocab_tgt, device)

            # Do a forward pass and compute the validation loss of this batch.
            logits, _ = model(x_in, seq_mask_x, seq_len_x, y_in)
            batch_NLL = model.loss(logits, y_out, reduction="sum")["loss"]
            val_NLL += batch_NLL.item()

            num_sentences += x_in.size(0)
            num_predictions += seq_len_y.sum().item()

    val_perplexity = np.exp(val_NLL / num_predictions)
    return val_perplexity, val_NLL/num_sentences
