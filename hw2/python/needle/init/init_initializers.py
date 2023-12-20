import math
from .init_basic import *


def xavier_uniform(fan_in: int, fan_out: int, gain=1.0, **kwargs):
    a = math.sqrt(6. / (fan_in + fan_out)) * gain
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    std = gain * math.sqrt(2. / (fan_in + fan_out))
    return randn(fan_in, fan_out, mean=0, std=std, **kwargs)


def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    gain = math.sqrt(2)  # the recommended gain for relu.
    a = gain * math.sqrt(3. / fan_in)
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    gain = math.sqrt(2)  # the recommended gain for relu.
    std = gain * math.sqrt(1. / fan_in)
    return randn(fan_in, fan_out, mean=0, std=std, **kwargs)
