import torch.nn as nn
from torch.distributions import Distribution


class Conditioner(nn.Module):
    """
    Use this to condition on input features and parameterise a batch of distributions over D-dimensional events.

    A distribution returned by the forward method is a torch.distribution.Distribution 
    and it should be such that its batch_shape is B and its event_shape is D.

    If you use a factorised distribution, e.g. product of D independent Gaussians, make sure to wrap them around
    torch.distribution.Independent, otherwise use a multivariate distribution, e.g. MultivariateNormal. 
    """

    def __init__(self):
        super().__init__()
    
    def forward(self, input_features) -> Distribution:        
        """Return a distribution parameterised by compute_parameters(input_features)."""
        raise NotImplementedError("How can I map from input features to a torch Distribution without code?!")

    def compute_parameters(self, input_features):
        """Return a list of parameters necessary for the distribution returned by the forward method"""
        raise NotImplementedError("How can I map from input features to parameters without code?!")

