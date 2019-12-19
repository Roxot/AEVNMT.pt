import torch.distributions as torchd
import probabll.distributions as probd

from .priors import PriorLayer, ProductOfPriorsLayer, GaussianPriorLayer, BetaPriorLayer, MoGPriorLayer, HardUniformPriorLayer
from .conditioner import Conditioner
from .normal import NormalLayer
from .kumaraswamy import KumaraswamyLayer, HardKumaraswamyLayer
from .mixture import MixtureOfGaussiansLayer
from .product import ProductOfConditionalsLayer


def get_named_params(p):
    """Return a sequence of pairs (param_name: str, param_value: tensor) based on the type of the distribution p"""
    return probd.get_named_params(p)

def create_prior(family, latent_size, params) -> PriorLayer:
    """Helper to create a prior module"""
    family = family.strip().lower()
    if family == "gaussian":
        if not params:
            params = [0., 1.]
        if len(params) != 2:
            raise ValueError("Specify the Gaussian prior using a location and a strictly positive scale.")
        return GaussianPriorLayer(latent_size, params[0], params[1])
    elif family == "beta":
        if not params:
            params = [0.5, 0.5]
        if len(params) != 2:
            raise ValueError("Specify the Beta prior using two strictly positive shape parameters.")
        return BetaPriorLayer(latent_size, params[0], params[1]) 
    elif family == "harduniform":
        if not params:
            params = [0.4, 0.5]
        if len(params) != 2:
            raise ValueError("Specify the HardUniform prior using two strictly positive shape parameters that add to less than 1.")
        return HardUniformPriorLayer(latent_size, params[0], params[1])
    elif family == "mog":
        if not params:
            params = [10, 10, 0.5]
        if len(params) != 3:
            raise ValueError("Specify the MoG prior using a number of components, a radius (for initialisation), and a strictly positive scale.")
        num_components = int(params[0])
        if num_components <= 1:
            raise ValueError("An MoG prior requires more than 1 component.")
        radius = params[1]
        if radius <= 0:
            raise ValueError("Initialising the MoG prior takes a strictly positive radius.")
        scale = params[2]
        return MoGPriorLayer(latent_size, num_components, radius, scale)
    else:
        raise ValueError("I do not know how to create a %r prior" % family)


__all__ = [
    "create_prior",
    "PriorLayer",
    "ProductOfPriorsLayer",
    "GaussianPriorLayer",
    "BetaPriorLayer",
    "HardUniformPriorLayer",
    "MoGPriorLayer",
    "Conditioner", 
    "NormalLayer",
    "KumaraswamyLayer",
    "HardKumaraswamyLayer",
    "MixtureOfGaussiansLayer",
    "ProductOfConditionalsLayer",
    "get_named_params"]

