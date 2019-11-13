import torch.nn as nn
import torch.distributions as torchdist
from aevnmt.dist import Conditioner


class NormalLayer(Conditioner):
    """
    Parameterise a product of Normal distributions.
    """

    def __init__(self, input_dim, hidden_size, latent_size):
        super().__init__()
        self.loc_layer = nn.Sequential(nn.Linear(input_dim, hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size, latent_size))
        self.scale_layer = nn.Sequential(nn.Linear(input_dim, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, latent_size),
                                       nn.Softplus())

    def forward(self, input_features):
        """Compute loc and scale as a function of input_features and return Independent(Normal(loc, scale))"""
        loc, scale = self.compute_parameters(input_features)
        return torchdist.Independent(torchdist.Normal(loc=loc, scale=scale), 1)

    def compute_parameters(self, input_features):
        """Return loc in R^D and scale in R+^D"""
        loc = self.loc_layer(input_features)
        scale = self.scale_layer(input_features)
        return loc, scale


