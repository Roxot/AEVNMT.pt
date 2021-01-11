import math
import torch
from torch import nn
from torch.distributions import Categorical
from torch_two_sample import MMDStatistic

from .constraints import Constraint

def log_likelihood_loss(likelihood, targets, ll_fn, pad_idx, label_smoothing=0.):
    """
    Returns the log-likelihood for the given likelihood and targets.

    :param likelihood: likelihood Distribution.
    :param targets: Targets for log_prob.
    :param ll_fn: log_prob function, built-in Categorical log_prob is not used because there is no padding support.
    :param pad_idx: idx of padding tokens in targets
    :param label_smoothing: label smoothing factor between (0, 1), defaults to 0.
    """
    log_likelihood = ll_fn(likelihood, targets)
    if label_smoothing > 0.:
        smooth_loss = label_smoothing_loss(likelihood, targets, ignore_index=pad_idx)
        log_likelihood = (1 - label_smoothing) * log_likelihood + label_smoothing * smooth_loss.sum(-1)
    return log_likelihood

def label_smoothing_loss(likelihood: Categorical, target, ignore_index=None):
    """
    Returns the unweighted label smoothing loss component, as defined in:

    Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna.
    Rethinking the inception architecture for computer vision.

    :param likelihood: Categorical distribution, the likelihood parameterized by the model.
    :param target: Target labels
    :param ignore_index: This index from target is not included in the loss, defaults to None
    """
    smooth_loss = likelihood.logits.sum(-1)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        smooth_loss.masked_fill_(pad_mask, 0.)
    smooth_loss = (smooth_loss / likelihood.probs.size(-1))
    return smooth_loss

def mmd_loss(sample_1, sample_2):
    """Computes an unbiased estimate of the MMD between two distributions given a set of samples from both.

    Source: https://github.com/tom-pelsmaeker/deep-generative-lm/blob/master/model/base_decoder.py
    """
    mmd = MMDStatistic(max(2, sample_1.shape[0]), max(2, sample_2.shape[0]))
    if sample_1.shape[0] == 1:
        return mmd(sample_1.expand(2, -1), sample_2.expand(2, -1), [1. / sample_1.shape[1]])
    else:
        return mmd(sample_1, sample_2, [1. / sample_1.shape[1]])


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_samples = 1


class LogLikelihoodLoss(Loss):
    def __init__(self, label_smoothing=0.):
        """
        A log-likelihood loss with label smoothing, for the NMT model without latent variables.

        :param label_smoothing: [description], defaults to 0.
        :type label_smoothing: [type], optional
        """
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, likelihood, targets, model, reduction="mean"):
        out_dict = dict()
        log_likelihood = log_likelihood_loss(likelihood, targets,
                                             model.translation_model.log_prob,
                                             model.translation_model.tgt_embedder.padding_idx,
                                             self.label_smoothing)
        loss = - log_likelihood

        if reduction == "mean":
            out_dict['loss'] = loss.mean()
        elif reduction == "sum":
            out_dict['loss'] = loss.sum()
        elif reduction == "none" or reduction is None:
            out_dict['loss'] = loss
        else:
            raise Exception(f"Unknown reduction option {reduction}")

        return out_dict


class ELBOLoss(Loss):
    def __init__(self, kl_weight=1., kl_annealing_steps=0., free_nats=0., mmd_weight=0., label_smoothing_x=0., label_smoothing_y=0.):
        """
        ELBOLoss implements both the ELBO [1] and InfoVAE loss for AEVNMT, by adding a mmd_weight to the regular ELBO.
        It is advised to use InfoVAELoss with the InfoVAE, as this class determines the proper kl_weight and mmd_weight from the InfoVAE parameters.

        [1] Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes."

        :param kl_weight: Weight of the KL term, defaults to 1.
        :param kl_annealing_steps: Adds a linear annealing schedule between [0, kl_weight], defaults to 0.
        :param free_nats: Minimum KL value, the KL is not optimized if below this value, defaults to 0.
        :param mmd_weight: Weight of the MMD term used with InfoVAE, defaults to 0.
        :param label_smoothing_x: source language label smoothing, defaults to 0.
        :param label_smoothing_y: target language label smoothing, defaults to 0.
        """
        super().__init__()
        self.kl_weight = kl_weight
        self.kl_annealing_steps = kl_annealing_steps
        self.free_nats = free_nats
        self.mmd_weight = mmd_weight
        self.label_smoothing_x = label_smoothing_x
        self.label_smoothing_y = label_smoothing_y

    def forward(self, tm_likelihood, lm_likelihood, targets_y, targets_x, qz, pz, sample_qz, step, model, reduction='mean'):
        """
        Computes an estimate of the negative evidence lower bound for the single sample of the latent
        variable that was used to compute the categorical parameters, and the distributions qz
        that the sample comes from.

        :param tm_likelihood: Categorical distributions from LM with shape [B, Ty, Vy]
        :param lm_likelihood: Categorical distributions from TM with shape [B, Tx, Vx]
        :param targets_y: target labels target sentence [B, T_y]
        :param targets_x: target labels source sentence [B, T_x]
        :param qz: distribution that was used to sample the latent variable.
        :param sample_z: The sample from q_z that was used to compute the likelihoods.
        :param model: The model that contains log_prob definitions to compute the likelihoods.
        :param reduction: what reduction to apply, none ([B]), mean ([]) or sum ([])
        """
        out_dict = dict()

        tm_log_likelihood=None
        if tm_likelihood is not None:
            tm_log_likelihood = log_likelihood_loss(tm_likelihood, targets_y,
                                                model.translation_model.log_prob,
                                                model.translation_model.tgt_embedder.padding_idx,
                                                self.label_smoothing_y)
        lm_log_likelihood = log_likelihood_loss(lm_likelihood, targets_x,
                                                model.language_model.log_prob,
                                                model.language_model.pad_idx,
                                                self.label_smoothing_x)

        # KL(q(z|x) || p(z))
        KL = torch.distributions.kl_divergence(qz, pz)
        raw_KL = KL * 1
        if self.free_nats > 0:
            KL = torch.clamp(KL, min=self.free_nats)
        KL_weight = self.kl_weight
        if self.kl_annealing_steps > 0:
            KL_weight = min(KL_weight, (KL_weight / self.kl_annealing_steps) * step)

        # ELBO and loss
        elbo = lm_log_likelihood - KL * KL_weight
        if tm_log_likelihood is not None:
            elbo+=tm_log_likelihood
        loss = - elbo

        # MMD(q(z) || p(z))
        if self.mmd_weight > 0.:
            sample_pz = pz.sample([sample_qz.size(0)])
            mmd = mmd_loss(sample_qz, sample_pz)
            out_dict['MMD'] = mmd
            # NOTE the loss is the negative ELBO,
            # -(ELBO + MMD) == - ELBO - MMD
            loss = loss - mmd * self.mmd_weight

        # Constraints
        if 'MDR' in model.constraints:
            constraint = model.constraints['MDR']
            mdr_loss = constraint(KL)
            loss = loss + mdr_loss
            out_dict[f'constraints/multipliers/{constraint.name}'] = constraint.multiplier.detach()

        if reduction == "mean":
            out_dict['loss'] = loss.mean()
        elif reduction == "sum":
            out_dict['loss'] = loss.sum()
        elif reduction == "none" or reduction is None:
            out_dict['loss'] = loss
        else:
            raise Exception(f"Unknown reduction option {reduction}")

        out_dict['KL'] = KL.detach()
        out_dict['raw_KL'] = raw_KL.detach()
        out_dict['lm/main'] = lm_log_likelihood.detach()
        if tm_log_likelihood is not None:
            out_dict['tm/main'] = tm_log_likelihood.detach()

        return out_dict


class InfoVAELoss(ELBOLoss):
    def __init__(self, info_alpha=1., info_lambda=1., kl_annealing_steps=0., free_nats=0., mmd_weight=0., label_smoothing_x=0., label_smoothing_y=0.):
        """
        The InfoVAE objective [2]. As this loss is fully covered by the ELBO implementation,
        the init only determines the correct KL and MMD weights from the InfoVAE parameters.

        [2] Zhao, Shengjia, Jiaming Song, and Stefano Ermon. "Infovae: Information maximizing variational autoencoders."
        """
        self.info_alpha = info_alpha
        self.info_lambda = info_lambda
        kl_weight = 1 - info_alpha
        mmd_weight = info_alpha + info_lambda - 1
        super().__init__(kl_weight, kl_annealing_steps, free_nats, mmd_weight, label_smoothing_x, label_smoothing_y)


class LagVAELoss(Loss):
    def __init__(self, alpha, label_smoothing_x=0., label_smoothing_y=0.):
        """
        LagVAE Loss [3]. The bounds for the ELBO and MMD constraints are defined when the constraints are constructed, in aevnmt_helper.py.

        [3] Zhao, Shengjia, Jiaming Song, and Stefano Ermon. "A lagrangian perspective on latent variable generative models."

        :param alpha: The scaling parameter that determines which bound is optimized.
        """
        super().__init__()
        self.alpha = alpha
        if alpha >= 0:
            raise NotImplementedError(f"Minimizing the MI upper bound is not implemented (alpha = {alpha}).")

        self.label_smoothing_x = label_smoothing_x
        self.label_smoothing_y = label_smoothing_y

    def forward(self, tm_likelihood, lm_likelihood, targets_y, targets_x, qz, pz, sample_qz, step, model, reduction='mean'):
        out_dict = dict()

        ll_py=None
        if tm_likelihood is not None:
            ll_py = log_likelihood_loss(tm_likelihood, targets_y,
                                    model.translation_model.log_prob,
                                    model.translation_model.tgt_embedder.padding_idx,
                                    self.label_smoothing_y)
        ll_px = log_likelihood_loss(lm_likelihood, targets_x,
                                    model.language_model.log_prob,
                                    model.language_model.pad_idx,
                                    self.label_smoothing_x)
        ll_total = ll_px
        if ll_py is not None:
            ll_total+=ll_py
        KL = torch.distributions.kl_divergence(qz, pz)
        neg_elbo = KL - ll_total
        mmd = mmd_loss(sample_qz, pz.sample([sample_qz.size(0)]))

        # Main objective
        loss = - ll_total

        # Add Constraints
        loss = loss + model.constraints['ELBO'](neg_elbo) + model.constraints['MMD'](mmd)

        if reduction == "mean":
            out_dict['loss'] = loss.mean()
        elif reduction == "sum":
            out_dict['loss'] = loss.sum()
        elif reduction == "none" or reduction is None:
            out_dict['loss'] = loss
        else:
            raise Exception(f"Unknown reduction option {reduction}")

        out_dict['raw_KL'] = KL.detach()
        if ll_py is not None:
            out_dict['tm/main'] = ll_py.detach()
        out_dict['lm/main'] = ll_px.detach()
        out_dict['mmd'] = mmd.detach()

        return out_dict


class IWAELoss(Loss):
    def __init__(self, num_samples, label_smoothing_x=0., label_smoothing_y=0.):
        """
        The Importance Weighted Autoencoder (IWAE) objective [4].

        [4] Burda, Yuri, Roger Grosse, and Ruslan Salakhutdinov. "Importance weighted autoencoders."

        :param num_samples: [description]
        :type num_samples: [type]
        :param label_smoothing_x: [description], defaults to 0.
        :type label_smoothing_x: [type], optional
        :param label_smoothing_y: [description], defaults to 0.
        :type label_smoothing_y: [type], optional
        """
        super().__init__()
        self.num_samples = num_samples
        self.label_smoothing_x = label_smoothing_x
        self.label_smoothing_y = label_smoothing_y

    def forward(self, tm_likelihood, lm_likelihood, targets_y, targets_x, qz, pz, sample_qz, step, model, reduction='mean'):
        out_dict = dict()
        batch_size = sample_qz.shape[0] // self.num_samples

        # log probabilities [num_samples, batch_size]
        ll_py=None
        if tm_likelihood is not None:
            ll_py = log_likelihood_loss(tm_likelihood, targets_y,
                                    model.translation_model.log_prob,
                                    model.translation_model.tgt_embedder.padding_idx,
                                    self.label_smoothing_y)
            ll_py = ll_py.view(self.num_samples, batch_size)
        ll_px = log_likelihood_loss(lm_likelihood, targets_x,
                                    model.language_model.log_prob,
                                    model.language_model.pad_idx,
                                    self.label_smoothing_x)
        ll_px = ll_px.view(self.num_samples, batch_size)

        sample_qz = sample_qz.view(self.num_samples, batch_size, -1)
        logprob_qz = qz.log_prob(sample_qz)
        logprob_pz = pz.log_prob(sample_qz)

        # Importance weights
        log_w = ll_px + logprob_pz - logprob_qz
        if ll_py is not None:
            log_w += ll_py
        raw_loss = torch.logsumexp(log_w, dim=0) - math.log(self.num_samples)

        # Loss with normalized importance weights (See Burda et al. (14))
        w_norm = torch.softmax(log_w, dim=0)
        loss = - torch.sum(w_norm.detach() * log_w, dim=0)

        if reduction == "mean":
            out_dict['loss'] = loss.mean()
        elif reduction == "sum":
            out_dict['loss'] = loss.sum()
        elif reduction == "none" or reduction is None:
            out_dict['loss'] = loss
        else:
            raise Exception(f"Unknown reduction option {reduction}")

        with torch.no_grad():
            out_dict['raw_KL'] = torch.distributions.kl_divergence(qz, pz)
            out_dict['IWAE_loss'] = raw_loss.detach()
            if ll_py is not None:
                out_dict['tm/main'] = torch.logsumexp(ll_py + logprob_pz - logprob_qz, dim=0)
            out_dict['lm/main'] = torch.logsumexp(ll_px + logprob_pz - logprob_qz, dim=0)

        return out_dict
