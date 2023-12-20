"""The module.
"""
from typing import List, Callable, Any, Optional
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
            self, in_features, out_features, bias=True, device=None,
            dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = init.kaiming_uniform(in_features, out_features)
        self.bias: Optional[Tensor] = None
        if bias:
            self.bias = init.kaiming_uniform(out_features, 1).transpose()

    def forward(self, X: Tensor) -> Tensor:
        result = X @ self.weight
        if self.bias:
            result += self.bias.broadcast_to((X.shape[0], self.out_features))
        return result


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        num_examples, num_classes = logits.shape
        y = init.one_hot(num_classes, y)
        return (
                ops.logsumexp(logits, (1,)).sum() - (logits * y).sum()
        ) / num_examples


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None,
                 dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.running_mean = init.zeros(dim)
        self.running_var = init.ones(dim)
        self.weight = init.ones(dim)
        self.bias = init.zeros(dim)

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[1] == self.dim
        assert len(x.shape) == 2
        if not self.training:
            mean = self.running_mean.broadcast_to(x.shape)
            var = self.running_var.broadcast_to(x.shape)
            normalized = (x - mean) / (var + self.eps) ** .5
            weight = self.weight.broadcast_to(x.shape).data
            bias = self.bias.broadcast_to(x.shape).data
            return normalized.data * weight + bias

        # if training:
        mean, var, normalized = normalize(x, self.eps, 0)
        self.running_mean += self.momentum * (mean.data - self.running_mean)
        self.running_var += self.momentum * (var.data - self.running_var)
        weight = self.weight.broadcast_to(x.shape)
        bias = self.bias.broadcast_to(x.shape)
        return normalized * weight + bias




class LayerNorm1d(Module):

    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = init.ones(dim)
        self.bias = init.zeros(dim)

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[1] == self.dim
        assert len(x.shape) == 2
        _, _, normalized = normalize(x, self.eps, 1)
        weight = self.weight.broadcast_to(x.shape)
        bias = self.bias.broadcast_to(x.shape)
        return weight * normalized + bias


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        gain = init.randb(*x.shape, p=self.p) / (1. - self.p)
        return x * gain



class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)


def normalize(x: Tensor, eps: float, axis: int):
    dim = x.shape[axis]
    axes = (axis,)
    keepdim_shape = ops.get_squashed_shape(axes, x.shape)
    mean = x.sum(axes) / dim
    broadcast_mean = mean.reshape(keepdim_shape).broadcast_to(x.shape)
    var = ((x - broadcast_mean) ** 2).sum(axes) / dim
    broadcast_var = var.reshape(keepdim_shape).broadcast_to(x.shape)
    normalized = (x - broadcast_mean) / (broadcast_var + eps) ** .5
    return mean, var, normalized


def keepdim_sum(x: Tensor, axes: Optional[tuple] = None):
    keepdim_shape = ops.get_squashed_shape(axes, x.shape)
    return x.sum(axes).reshape(keepdim_shape)
