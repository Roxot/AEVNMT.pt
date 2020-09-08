import functools
from argparse import ArgumentTypeError


def str_to_str_list(values):
    if isinstance(values, str):
        values = (str(v) for v in values.split())
    else:
        values = (str(v) for v in values)
    return list(values)


def str_to_float_list(values):
    if isinstance(values, str):
        values = (float(v) for v in values.split())
    else:
        values = (float(v) for v in values)
    return list(values)


def str_to_int_list(values):
    if isinstance(values, str):
        values = (int(v) for v in values.split())
    else:
        values = (int(v) for v in values)
    return list(values)


def str_to_bool(v):
    """Source: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise ArgumentTypeError('Boolean value expected.')


def rsetattr(obj, attr, val):
    """
    setattr for nested attributes. 
    Source: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
    """
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    """
    getattr for nested attributes.
    Source: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
    """
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))
