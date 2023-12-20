from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api


class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        z_max = array_api.max(Z, axis=self.axes, keepdims=True)

        return array_api.squeeze(array_api.log(
            array_api.sum(
                array_api.exp(Z - z_max),
                axis=self.axes, keepdims=True,
            ),
        ) + z_max, axis=self.axes)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        Z = node.inputs[0]
        assert isinstance(Z, Tensor)
        # loop hole, the chain of gradients break here.
        z_max = Z.realize_cached_data().max(self.axes, keepdims=True)
        exp_z = exp(Z - Tensor(z_max).broadcast_to(Z.shape))
        shape = get_squashed_shape(self.axes, Z.shape)
        grad_sum_exp_z = ((out_grad/ exp_z.sum(self.axes))
                          .reshape(shape)
                          .broadcast_to(Z.shape))
        # TODO(zhangfan): is using realize_cached_data a good idea?
        return exp_z * grad_sum_exp_z


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
