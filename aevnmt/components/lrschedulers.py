"""
    The NoamScheduler here is from: https://github.com/joeynmt/joeynmt/blob/master/joeynmt/builders.py
"""

import torch

class NoamScheduler:
    """
    The Noam learning rate scheduler used in "Attention is all you need"
    See Eq. 3 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, hidden_size: int, optimizer: torch.optim.Optimizer,
                 factor: float = 1, warmup: int = 4000):
        """
        Warm-up, followed by learning rate decay.
        :param hidden_size:
        :param optimizer:
        :param factor: decay factor
        :param warmup: number of warmup steps
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.hidden_size = hidden_size
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self._compute_rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate

    def _compute_rate(self):
        """Implement `lrate` above"""
        step = self._step
        return self.factor * \
            (self.hidden_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    #pylint: disable=no-self-use
    def state_dict(self):
        return None

