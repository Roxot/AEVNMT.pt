from pathlib import Path
import sys
import torch
import torch.optim as optim
import numpy as np
import sacrebleu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from aevnmt.data import MemMappedCorpus, MemMappedParallelCorpus
from aevnmt.data import Vocabulary, ParallelDataset, TextDataset, remove_subword_tokens
from aevnmt.components import BahdanauAttention, BahdanauDecoder, LuongAttention, LuongDecoder, TransformerEncoder, RNNEncoder

def load_data(hparams, vocab_src, vocab_tgt, use_memmap=False):
    train_src = f"{hparams.training_prefix}.{hparams.src}"
    train_tgt = f"{hparams.training_prefix}.{hparams.tgt}"
    val_src = f"{hparams.validation_prefix}.{hparams.src}"
    val_tgt = f"{hparams.validation_prefix}.{hparams.tgt}"
    opt_data = dict()

    if use_memmap:
        print('Memory mapping bilingual data')
        training_data = MemMappedParallelCorpus(
            [train_src, train_tgt],
            f"{hparams.output_dir}/training.memmap",
            [vocab_src, vocab_tgt],
            max_length=hparams.max_sentence_length
        )
        # Generally there's no need for memory mapping validation data
        print('Loading validation data')
        val_data = ParallelDataset(val_src, val_tgt, max_length=-1)

        if hparams.mono_src:
            print('Memory mapping source monolingual data')
            opt_data['mono_src'] = MemMappedCorpus(
                hparams.mono_src,
                f"{hparams.output_dir}/mono_src.memmap",
                vocab_src,
                max_length=hparams.max_sentence_length
            )
        if hparams.mono_tgt:
            print('Memory mapping target monolingual data')
            opt_data['mono_tgt'] = MemMappedCorpus(
                hparams.mono_tgt,
                f"{hparams.output_dir}/mono_tgt.memmap",
                vocab_tgt,
                max_length=hparams.max_sentence_length
            )
    else:
        training_data = ParallelDataset(train_src, train_tgt, max_length=hparams.max_sentence_length)
        val_data = ParallelDataset(val_src, val_tgt, max_length=-1)
        if hparams.mono_src:
            opt_data['mono_src'] = TextDataset(hparams.mono_src, max_length=hparams.max_sentence_length)
        if hparams.mono_tgt:
            opt_data['mono_tgt'] = TextDataset(hparams.mono_tgt, max_length=hparams.max_sentence_length)

    return training_data, val_data, opt_data

def load_vocabularies(hparams):
    train_src = f"{hparams.training_prefix}.{hparams.src}"
    train_tgt = f"{hparams.training_prefix}.{hparams.tgt}"
    val_src = f"{hparams.validation_prefix}.{hparams.src}"
    val_tgt = f"{hparams.validation_prefix}.{hparams.tgt}"

    # Construct the vocabularies.
    if hparams.vocab_prefix is not None:

        if hparams.share_vocab:
            vocab = Vocabulary.from_file(hparams.vocab_prefix, max_size=hparams.max_vocab_size)
            vocab_src = vocab
            vocab_tgt = vocab
        else:
            vocab_src_file = f"{hparams.vocab_prefix}.{hparams.src}"
            vocab_tgt_file = f"{hparams.vocab_prefix}.{hparams.tgt}"
            vocab_src = Vocabulary.from_file(vocab_src_file, max_size=hparams.max_vocab_size)
            vocab_tgt = Vocabulary.from_file(vocab_tgt_file, max_size=hparams.max_vocab_size)
    else:

        if hparams.share_vocab:
            all_files = [train_src, val_src, train_tgt, val_tgt]
            if hparams.mono_src:
                all_files.append(hparams.mono_src)
            if hparams.mono_tgt:
                all_files.append(hparams.mono_tgt)
            vocab = Vocabulary.from_data(all_files,
                                         min_freq=hparams.vocab_min_freq, max_size=hparams.max_vocab_size)
            vocab_src = vocab
            vocab_tgt = vocab
        else:
            src_files = [train_src, val_src]
            if hparams.mono_src:
                src_files.append(hparams.mono_src)
            vocab_src = Vocabulary.from_data(src_files, min_freq=hparams.vocab_min_freq,
                                             max_size=hparams.max_vocab_size)
            tgt_files = [train_tgt, val_tgt]
            if hparams.mono_tgt:
                tgt_files.append(hparams.mono_tgt)
            vocab_tgt = Vocabulary.from_data(tgt_files, min_freq=hparams.vocab_min_freq,
                                             max_size=hparams.max_vocab_size)

    return vocab_src, vocab_tgt

def create_encoder(hparams):
    if hparams.encoder_style == "rnn":
        return RNNEncoder(emb_size=hparams.emb_size,
                             hidden_size=hparams.hidden_size,
                             bidirectional=hparams.bidirectional,
                             dropout=hparams.dropout,
                             num_layers=hparams.num_enc_layers,
                             cell_type=hparams.cell_type)
    elif hparams.encoder_style == "transformer":
        return TransformerEncoder(input_size=hparams.emb_size,
                                     num_heads=hparams.transformer_heads,
                                     num_layers=hparams.num_enc_layers,
                                     dim_ff=hparams.transformer_hidden,
                                     dropout=hparams.dropout)
    else:
        raise Exception(f"Unknown encoder style: {hparams.encoder_style}")

def create_decoder(attention, hparams):
    init_from_encoder_final = (hparams.model_type == "cond_nmt")
    if hparams.decoder_style == "bahdanau":
        return BahdanauDecoder(emb_size=hparams.emb_size,
                               hidden_size=hparams.hidden_size,
                               attention=attention,
                               dropout=hparams.dropout,
                               num_layers=hparams.num_dec_layers,
                               cell_type=hparams.cell_type,
                               init_from_encoder_final=init_from_encoder_final)
    elif hparams.decoder_style == "luong":
        return LuongDecoder(emb_size=hparams.emb_size,
                            hidden_size=hparams.hidden_size,
                            attention=attention,
                            dropout=hparams.dropout,
                            num_layers=hparams.num_dec_layers,
                            cell_type=hparams.cell_type,
                            init_from_encoder_final=init_from_encoder_final)
    else:
        raise Exception(f"Unknown decoder style: {hparams.decoder_style}")

def create_attention(hparams):
    if not hparams.attention in ["luong", "scaled_luong", "bahdanau"]:
        raise Exception(f"Unknown attention option: {hparams.attention}")

    if hparams.encoder_style == "rnn":
        key_size = hparams.hidden_size
        if hparams.bidirectional:
            key_size = key_size * 2
    else:
        key_size = hparams.emb_size
    query_size = hparams.hidden_size

    if "luong" in hparams.attention:
        scale = True if hparams.attention == "scaled_luong" else False
        attention = LuongAttention(key_size, hparams.hidden_size, scale=scale)
    else:
        attention = BahdanauAttention(key_size, query_size, hparams.hidden_size)

    return attention

def model_parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parameter_count(param_generator):
    return sum(p.numel() for p in param_generator if p.requires_grad)

def gradient_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = np.sqrt(total_norm)
    return total_norm

def attention_summary(src_labels, tgt_labels, att_weights, summary_writer, summary_name,
                      global_step):
    """
    :param src_labels: [T_src] source word tokens (strings)
    :param tgt_labels: [T_tgt] target word tokens (strings)
    :param att_weights: [T_tgt, T_src]
    """

    # Plot a heatmap for the scores.
    fig, ax = plt.subplots()
    plt.imshow(att_weights, cmap="Greys", aspect="equal",
               origin="upper", vmin=0., vmax=1.)

   # Configure the columns.
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(att_weights.shape[1]))
    ax.set_xticklabels(tgt_labels, rotation="vertical")

    # Configure the rows.
    ax.set_yticklabels(src_labels)
    ax.set_yticks(np.arange(att_weights.shape[0]))

    # Fit the figure neatly.
    plt.tight_layout()

    # Write the summary.
    summary_writer.add_figure(summary_name, fig, global_step=global_step)

def compute_bleu(hypotheses, references, subword_token=None):
    """
    Computes sacrebleu for a single set of references.
    """

    # Remove any subword tokens such as "@@".
    if subword_token is not None:
        references = remove_subword_tokens(references, subword_token)
        hypotheses = remove_subword_tokens(hypotheses, subword_token)

    # Compute the BLEU score.
    return sacrebleu.raw_corpus_bleu(hypotheses, [references]).score

class StepCounter:

    def __init__(self):
        self.total = 0
        self.mono_src = 0
        self.mono_tgt = 0
        self.bilingual = 0

    def count(self, batch_type):
        self.total += 1
        if batch_type == 'xy' or batch_type == 'yx':
            self.bilingual += 1
        elif batch_type == 'x':
            self.mono_src += 1
        elif batch_type == 'y':
            self.mono_tgt += 1
        else:
            raise ValueError("Unknown batch type: %s" % batch_type)

    def step(self, batch_type=None):
        if batch_type is None:
            return self.total
        if batch_type == 'xy' or batch_type == 'yx':
            return self.bilingual
        elif batch_type == 'x':
            return self.mono_src
        elif batch_type == 'y':
            return self.mono_tgt
        else:
            raise ValueError("Unknown batch type: %s" % batch_type)


class CheckPoint:
    """
    Use this class to save/load checkpoints with respect to utility metrics.
    """

    def __init__(self, model_dir: Path, metrics: list):
        self._metrics = metrics
        self._data = {metric: {
            "value": -float('inf'),
            "step": 0,
            "epoch": 0,
            "no_improvement": 0,
            "dir": model_dir/metric,
            "log": model_dir/metric/"log"
        } for metric in metrics}
        for data in self._data.values():
            data["dir"].mkdir(parents=True, exist_ok=True)

    def change_model_dir(self, model_dir: Path):
        for metric, data in self._data.items():
            data["dir"] = model_dir/metric
            data["dir"].mkdir(parents=True, exist_ok=True)
            data["log"] = model_dir/metric/"log"

    def update(self, epoch, step, models: dict, **kwargs):
        """
        :param epoch:
        :param step:
        :param models: dictionary mapping a name to a model, e.g.
            {f"{src}-{tgt}": model_src_tgt, f"{tgt}-{src}": model_tgt_src}
        :param kwargs:
        :return:
        """
        for k, v in kwargs.items():
            data = self._data[k]
            # Save the best model wrt BLEU
            if v > data["value"]:
                data["no_improvement"] = 0
                data["value"] = v
                data["epoch"] = epoch
                data["step"] = step
                for name, model in models.items():
                    torch.save(model.state_dict(), data["dir"] / f"{name}.pt")
                with open(data["log"], 'a+') as f:
                    print(f"epoch={epoch} step={step} value={v}", file=f)
                #for name, opt in optimizers.items():
                #    torch.save(opt.state_dict(), data["dir"] / f"{name}.pt")
            else:
                data["no_improvement"] += 1

    def no_improvement(self, metric):
        return self._data[metric]["no_improvement"]

    def load_best(self, models: dict, metric: str):
        data = self._data[metric]
        for name, model in models.items():
            model.load_state_dict(torch.load(data["dir"] / f"{name}.pt"))
        return data
