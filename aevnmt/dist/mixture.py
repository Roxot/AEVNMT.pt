import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution
from probabll.distributions import MixtureOfGaussians

from aevnmt.dist import Conditioner


class MixtureOfGaussiansLayer(Conditioner):

    def __init__(self, input_dim, hidden_size, latent_size, num_components):
        super().__init__()
        # TODO: should I use something more expressive than this?
        self.latent_size = latent_size
        self.loc_layer = nn.Sequential(nn.Linear(input_dim, hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size, latent_size * num_components))
        self.scale_layer = nn.Sequential(nn.Linear(input_dim, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, latent_size * num_components),
                                       nn.Softplus())
        self.logits_layer = nn.Sequential(nn.Linear(input_dim, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, num_components))

    def forward(self, input_features) -> Distribution:
        # [..., K], [..., K*D], [..., K*D]
        logits, locations, scales = self.compute_parameters(input_features)
        # K times [..., D]
        locations = torch.split(locations, self.latent_size, dim=-1)
        # [..., K, D]
        locations = torch.stack(locations, dim=-2)
        # K times [..., D]
        scales = torch.split(scales, self.latent_size, dim=-1)
        # [..., K, D]
        scales = torch.stack(scales, dim=-2)
        return MixtureOfGaussians(logits=logits, locations=locations, scales=scales)

    def compute_parameters(self, input_features):
        logits = self.logits_layer(input_features)
        loc = self.loc_layer(input_features)
        scale = self.scale_layer(input_features)
        return logits, loc, scale

