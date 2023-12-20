"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

import numpy as array_api


# NOTE: implicit broadcasting is not supported yet.
# We detect incorrectly relying on implicit broadcasting by shape checking.

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        assert a.shape == b.shape, f"a={a.shape}, b={b.shape}"
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        assert isinstance(scalar, (int, float)), scalar
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        assert a.shape == b.shape, f"a={a.shape}, b={b.shape}"
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        a = node.inputs[0]
        return array_api.multiply(out_grad,
                                  a ** (self.scalar - 1) * self.scalar)


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        assert a.shape == b.shape, f"a={a.shape}, b={b.shape}"
        return a ** b

    def gradient(self, out_grad: Tensor, node: Tensor):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
                node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a ** b) * array_api.log(a.data)
        return grad_a, grad_b


def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a: NDArray, b: NDArray):
        assert a.shape == b.shape, f"a={a.shape}, b={b.shape}"
        return a / b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad / rhs, - out_grad * lhs / (rhs ** 2.)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a / self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    """Reverses the order of two axes (axis1, axis2), defaults to the last two axes."""

    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a: NDArray):
        dims = len(a.shape)
        i, j = self.axes if self.axes else (dims - 2, dims - 1)
        indices = list(range(dims))
        indices[i], indices[j] = indices[j], indices[i]
        return array_api.transpose(a, indices)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad.transpose(self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    """Give a new shape to an array without changing its data."""

    def __init__(self, shape: tuple):
        self.shape = shape

    def compute(self, a: NDArray):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        a = node.inputs[0]
        assert isinstance(a, Tensor)
        return out_grad.reshape(a.shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    """Broadcast an array to a new shape"""

    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        a = node.inputs[0]
        assert isinstance(a, Tensor)
        axes = get_broadcast_axes(a.shape, out_grad.shape)
        # TODO(zhangfan): Not a fan of it since it feels hacky.
        #   the reshape is needed since axes can remove the dimensions
        #   that suppose to exists. the proper fix would be to keep 1-sized dims.
        return out_grad.sum(tuple(axes)).reshape(a.shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    """sum of array elements over given axes"""

    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a: NDArray):
        return array_api.sum(a, self.axes)

    def gradient(self, out_grad: Tensor, node: Tensor):
        a = node.inputs[0]
        assert isinstance(a, Tensor)

        squashed_shape = get_squashed_shape(self.axes, a.shape)
        return out_grad.reshape(squashed_shape).broadcast_to(a.shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        assert len(a.shape) == len(b.shape)
        assert a.shape[:-2] == b.shape[:-2]
        assert a.shape[-1] == b.shape[-2]
        return a @ b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        assert isinstance(lhs, Tensor)
        assert isinstance(rhs, Tensor)

        # lhs[i, j] can affect out_grad[i, k] for any k;
        # for a given k, the multiplier is rhs[j, k].
        # so lhs[i,j] = sum_k out_grad[i, k] * rhs[j, k]
        # in vector from: out_grad @ rhs.T
        #
        # the right-hand side can be derived similarly, or consider the
        # transpose of the equation since now rhs becomes lhs.
        lhs_grad = out_grad @ rhs.transpose()  # n x k @ k x m = n x m
        rhs_grad = lhs.transpose() @ out_grad  # m x n @ n x k = m x n

        lhs_grad = lhs_grad.sum(
            get_broadcast_axes(lhs.shape, lhs_grad.shape)).reshape(lhs.shape)
        rhs_grad = rhs_grad.sum(
            get_broadcast_axes(rhs.shape, rhs_grad.shape)).reshape(rhs.shape)

        return lhs_grad, rhs_grad


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a: NDArray):
        return -a

    def gradient(self, out_grad: Tensor, node: Tensor):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a: NDArray):
        return array_api.log(a)

    def gradient(self, out_grad: Tensor, node: Tensor):
        a = node.inputs[0]
        return out_grad / a


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a: NDArray):
        return array_api.exp(a)

    def gradient(self, out_grad: Tensor, node: Tensor):
        a = node.inputs[0]
        assert isinstance(a, Tensor)
        assert isinstance(out_grad, Tensor)
        assert isinstance(node, Tensor)

        print(a.shape, a.dtype)
        print(out_grad.shape, out_grad.dtype)
        return out_grad * exp(a)


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a: NDArray):
        return a * (a > 0)

    def gradient(self, out_grad: Tensor, node: Tensor):
        a = node.inputs[0]
        assert isinstance(a, Tensor)
        return out_grad * (a.realize_cached_data() > 0)


def relu(a):
    return ReLU()(a)


def get_broadcast_axes(old: tuple, new: tuple):
    """Return axes in the new shape that were expanded via broadcast given the
    old shape."""
    assert len(new) >= len(old)
    axes = []
    for i in range(len(new)):
        if i < len(old) and new[-(i + 1)] == old[-(i + 1)]:
            continue
        axes.append(len(new) - i - 1)

    return tuple(axes)


def get_squashed_shape(axes: Optional[tuple], shape: tuple) -> tuple:
    """Return the shape after squashing axes without reducing dimensions.
    If axes is None, default to all axes of the shape.
    """
    axes = axes if axes else range(len(shape))
    squashed = list(shape)
    for axis in axes:
        squashed[axis] = 1
    return tuple(squashed)
