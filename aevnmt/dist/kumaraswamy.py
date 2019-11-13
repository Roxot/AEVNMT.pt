import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Independent
from probabll.distributions import Kumaraswamy

from aevnmt.dist import Conditioner


class KumaraswamyLayer(Conditioner):
    """
    Parameterise a product of independent Kumaraswamy distributions.
    A [Kumaraswamy distribution](https://en.wikipedia.org/wiki/Kumaraswamy_distribution) is a two-parameters distribution
    that somewhat resembles a Beta distribution, though he Kumaraswamy is not symmetric for settings where a Beta would be. 
    Notably, a Kumaraswamy has a simple cdf whose inverse is known and therefore is amenable to a differentiable reparameterisation.

    It was first used in a VAE by https://arxiv.org/pdf/1605.06197.pdf
    """

    def __init__(self, input_dim, hidden_size, latent_size, min_shape=1e-3, max_shape=10):
        super().__init__()
        self.min_shape = min_shape
        self.max_shape = max_shape
        self.pre_a_layer = nn.Sequential(nn.Linear(input_dim, hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size, latent_size))
        self.pre_b_layer = nn.Sequential(nn.Linear(input_dim, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, latent_size))

    def forward(self, input_features):
        a, b = self.compute_parameters(input_features)
        return Independent(Kumaraswamy(a=a, b=b), 1)

    def compute_parameters(self, input_features):
        """
        Return shape parameters (a, b) in R+^D.

        Note we do constraint the shape parameters to (min_shape, max_shape) to avoid numerical instabilities, 
        esp in the KL approximation.
        """
        pre_a = self.pre_a_layer(input_features)
        pre_b = self.pre_b_layer(input_features)
        a = torch.clamp(self.min_shape + F.softplus(pre_a), max=self.max_shape)
        b = torch.clamp(self.min_shape + F.softplus(pre_b), max=self.max_shape)
        return a, b

