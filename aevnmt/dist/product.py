import torch
import torch.nn as nn
from torch.distributions import Distribution 

from probabll.distributions import ProductOfDistributions

from .conditioner import Conditioner


class ProductOfConditionalsLayer(Conditioner):

    def __init__(self, conditioners: 'List[Conditioner]'):
        super().__init__()
        self.conditioners = nn.ModuleList(conditioners)

    def forward(self, input_features) -> Distribution:
        distributions = self.compute_parameters(input_features)
        return ProductOfDistributions(distributions)

    def compute_parameters(self, input_features):
        """
        Return shape parameters (a, b) in R+^D.

        Note we do constraint the shape parameters to (min_shape, max_shape) to avoid numerical instabilities, 
        esp in the KL approximation.
        """
        return [conditioner(input_features) for conditioner in self.conditioners]

