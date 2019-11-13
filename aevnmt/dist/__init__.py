import torch.distributions as torchd
import probabll.distributions as probd

from .conditioner import Conditioner
from .normal import NormalLayer
from .kumaraswamy import KumaraswamyLayer
from .mixture import MixtureOfGaussiansLayer


def get_named_params(p):
    """Return a sequence of pairs (param_name: str, param_value: tensor) based on the type of the distribution p"""
    if isinstance(p, torchd.Normal):
        return ('loc', p.loc), ('scale', p.scale)
    elif isinstance(p, torchd.Beta):
        return ('a', p.concentration1), ('b', p.concentration0)
    elif isinstance(p, torchd.Independent):
        return get_named_params(p.base_dist)
    elif isinstance(p, probd.Kumaraswamy):
        return ('a', p.a), ('b', p.b)
    elif isinstance(p, probd.MixtureOfGaussians):
        return (('log_w', p.log_weights),) + get_named_params(p.components)
    else:
        return tuple()


__all__ = [
    "Conditioner", 
    "NormalLayer",
    "KumaraswamyLayer",
    "MixtureOfGaussiansLayer",
    "get_named_params"]

