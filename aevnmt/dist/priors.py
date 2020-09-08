import torch
import torch.nn as nn
from torch.distributions import Distribution, Independent, Normal, Beta, Uniform

from probabll.distributions import MixtureOfGaussians, ProductOfDistributions, MixtureD01C01


class PriorLayer(nn.Module):
    """
    Use this to instantiate a prior -- that is set it's parameters.
    A prior in AEVNMT is a multivariate distribution, it's batch_shape is [] and event_shape is [latent_size].
    """

    def __init__(self):
        super().__init__()


class GaussianPriorLayer(PriorLayer):

    def __init__(self, latent_size, loc=0., scale=1.):
        super().__init__()
        if scale <= 0:
            raise ValueError("The scale of a Gaussian distribution is strictly positive: %r" % params)
        self.register_buffer("loc", torch.full([latent_size], loc))
        self.register_buffer("scale", torch.full([latent_size], scale))

    def forward(self) -> Distribution:
        return Independent(torch.distributions.Normal(loc=self.loc, scale=self.scale), 1)


class BetaPriorLayer(PriorLayer):

    def __init__(self, latent_size, a=0.5, b=0.5):
        super().__init__()
        if a <= 0. or b <= 0.:
            raise ValueError("The shape parameters of a Beta distribution are strictly positive: %r" % params)
        self.register_buffer("a", torch.full([latent_size], a))
        self.register_buffer("b", torch.full([latent_size], b))

    def forward(self) -> Distribution:
        return Independent(Beta(self.a, self.b), 1)


class MoGPriorLayer(PriorLayer):

    def __init__(self, latent_size, num_components=10, radius=10, scale=0.5):
        super().__init__()
        if scale <= 0:
            raise ValueError("The prior variance must be strictly positive.")
        # uniform prior over components
        self.register_buffer("logits", torch.ones(num_components))
        self.register_buffer("locations", - radius + torch.rand([num_components, latent_size]) * 2 * radius)
        self.register_buffer("scales", torch.full([num_components, latent_size], scale))

    def forward(self) -> Distribution:
        p = MixtureOfGaussians(logits=self.logits, locations=self.locations, scales=self.scales)
        return p


class HardUniformPriorLayer(PriorLayer):

    def __init__(self, latent_size, p0=0.4, p1=0.4):
        super().__init__()
        if not (0. < p0 < 1. and 0. < p1 < 1. and p0 + p1 < 1.):
            raise ValueError("The shape parameters of the HardUniform distribution must be strictly positive and sum to less than 1: %r" % params)
        self.register_buffer("p0", torch.full([latent_size], p0))
        self.register_buffer("p1", torch.full([latent_size], p1))
        self.register_buffer("pc", torch.full([latent_size], 1. - p0 - p1))

    def forward(self) -> Distribution:
        cont = Uniform(torch.zeros_like(self.p0), torch.ones_like(self.p0))
        probs = torch.stack([self.p0, self.p1, self.pc], -1)
        return Independent(MixtureD01C01(cont, probs=probs), 1)


class ProductOfPriorsLayer(PriorLayer):
    """
    A product of priors layer instantiates a number of independent priors.
    """

    def __init__(self, priors: list):
        super().__init__()
        self.priors = nn.ModuleList(priors)

    def forward(self) -> Distribution:
        return ProductOfDistributions([prior() for prior in self.priors])

