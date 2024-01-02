from typing import Optional

import pytest

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api


class LogSoftmax(TensorOp):
    def compute(self, z: NDArray):
        return z - _logsumexp(z)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        z = node.inputs[0]
        assert isinstance(z, Tensor)
        z_max = Tensor(z.realize_cached_data().max(keepdims=True))
        exp_z = exp(z - z_max)
        sum_exp_z = exp_z.sum().reshape((1,)).broadcast_to(z.shape)
        return out_grad * (1.0 - exp_z / sum_exp_z)


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        return _logsumexp(Z, self.axes)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        Z = node.inputs[0]
        assert isinstance(Z, Tensor)
        # Note(zhangfan): using realize_cached_data means we won't be able to
        # compute higher order gradients.
        z_max = Z.realize_cached_data().max(self.axes, keepdims=True)
        exp_z = exp(Z - Tensor(z_max).broadcast_to(Z.shape))
        shape = get_squashed_shape(self.axes, Z.shape)
        grad_sum_exp_z = (
            (out_grad / exp_z.sum(self.axes)).reshape(shape).broadcast_to(Z.shape)
        )
        return exp_z * grad_sum_exp_z


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


def _logsumexp(z: NDArray, axes: Optional[tuple] = None):
    z_max = array_api.max(z, axis=axes, keepdims=True)

    return array_api.squeeze(
        array_api.log(
            array_api.sum(
                array_api.exp(z - z_max),
                axis=axes,
                keepdims=True,
            ),
        )
        + z_max,
        axis=axes,
    )
