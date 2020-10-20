import torch
from torch.distributions import Categorical


def label_smoothing_loss(likelihood: Categorical, target, ignore_index=None):
    """
    Returns the unweighted label smoothing loss component, as defined in [1].

    [1] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna.
    Rethinking the inception architecture for computer vision.CoRR, abs/1512.00567, 2015.

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
