import torch.nn as nn
import torch.distributions as torchdist

class NormalLayer(nn.Module):

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
        loc, scale = self.compute_parameters(input_features)
        return torchdist.normal.Normal(loc=loc, scale=scale)

    def compute_parameters(self, input_features):
        loc = self.loc_layer(input_features)
        scale = self.scale_layer(input_features)
        return loc, scale
