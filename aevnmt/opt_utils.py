import torch.optim as optim
import torch.nn as nn
from functools import partial
from itertools import tee

from aevnmt.components import NoamScheduler

class RequiresGradSwitch:
    """
    Use this to temporarily switch gradient computation on/off for a subset of parameters:
    1. first you requires_grad(value), this will set requires_grad flags to a chosen value (False/True)
        while saving the original value of the flags
    2. then restore(), this will restore requires_grad to its original value
        i.e. whatever requires_grad was before you used requires_grad(value)
    """

    def __init__(self, param_generator):
        self.parameters = param_generator
        self.flags = None

    def requires_grad(self, requires_grad):
        if self.flags is not None:
            raise ValueError("Must restore first")
        self.parameters, parameters = tee(self.parameters, 2)
        flags = []
        for param in parameters:
            flags.append(param.requires_grad)
            param.requires_grad = requires_grad
        self.flags = flags

    def restore(self):
        if self.flags is None:
            raise ValueError("Nothing to restore")
        self.parameters, parameters = tee(self.parameters, 2)
        for param, flag in zip(parameters, self.flags):
            param.requires_grad = flag
        self.flags = None


def get_optimizer(name, parameters, lr, l2_weight, momentum=0.):
    if name is None or name == "adam":
        cls = optim.Adam
    elif name == "amsgrad":
        cls = partial(optim.Adam, amsgrad=True)
    elif name == "adadelta":
        cls = optim.Adadelta
    elif name == "rmsprop":
        cls = partial(optim.RMSprop, momentum=momentum)
    elif name == 'sgd':
        cls = optim.SGD
    else:
        raise ValueError("Unknown optimizer: %s" % name)
    return cls(params=parameters, lr=lr, weight_decay=l2_weight)


def get_lr_scheduler(optimizer, hparams):
    if hparams.lr_scheduler == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=hparams.lr_reduce_factor,
            patience=hparams.lr_reduce_patience,
            verbose=False,
            threshold=1e-2,
            threshold_mode="abs",
            cooldown=hparams.lr_reduce_cooldown,
            min_lr=hparams.min_lr)
    elif hparams.lr_scheduler == "noam":
        scheduler = NoamScheduler(hidden_size=hparams.emb_size,
                                  optimizer=optimizer,
                                  factor=hparams.lr_reduce_factor,
                                  warmup=hparams.lr_warmup)
    return scheduler


def construct_optimizers(hparams, gen_parameters, inf_z_parameters):
    optimizers = {
        "gen": get_optimizer(
            hparams.gen_optimizer,
            gen_parameters,
            hparams.gen_lr,
            hparams.gen_l2_weight
        ),
    }

    if inf_z_parameters is not None:
        optimizers["inf_z"] = get_optimizer(
            hparams.inf_z_optimizer,
            inf_z_parameters,
            hparams.inf_z_lr,
            hparams.inf_z_l2_weight
        )

    lr_schedulers = {
        "gen": get_lr_scheduler(
            optimizers["gen"],
            hparams
        ),
    }

    if inf_z_parameters is not None:
        lr_schedulers["inf_z"] = get_lr_scheduler(
            optimizers["inf_z"],
            hparams
        )

    return optimizers, lr_schedulers


def lr_scheduler_step(lr_schedulers, hparams, val_score=None):
    """
    Only updates if it's appropriate for the scheduler. I.e. it updates the LRonPLateauScheduler
    only during validation, and the NoamScheduler only during training.
    """
    for name, lr_scheduler in lr_schedulers.items():

        if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):

            # Don't update the ReduceLROnPlateau scheduler during training.
            if val_score is None:
                continue

            lr_scheduler.step(val_score)
            if lr_scheduler.cooldown_counter == hparams.lr_reduce_cooldown:
                print(f"Reduced the learning rate for '{name}' with a factor"
                      f" {hparams.lr_reduce_cooldown}")

        if isinstance(lr_scheduler, NoamScheduler):

            # Don't update the NoamScheduler during validation.
            if val_score is not None:
                continue
            lr_scheduler.step()

def take_optimizer_step(optimizer, parameters, clip_grad_norm=0., zero_grad=True):
    if clip_grad_norm > 0:
        nn.utils.clip_grad_norm_(parameters=parameters,
                                 max_norm=clip_grad_norm,
                                 norm_type=float("inf"))
    optimizer.step()
    if zero_grad:
        optimizer.zero_grad()
