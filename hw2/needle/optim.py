"""Optimization module"""
from typing import List

import needle as ndl
import numpy as np

from needle.nn import Parameter


class Optimizer:
    def __init__(self, params: List[Parameter]):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(
        self, params: List[Parameter], lr=0.01, momentum=0.0, weight_decay=0.0
    ):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for p in self.params:
            assert isinstance(p, ndl.Tensor)
            grad = p.grad.data + self.weight_decay * p.data
            u = self.u.get(p, 0.0)
            u = self.momentum * u + (1 - self.momentum) * grad
            self.u[p] = u.detach()

            p.data = (p - self.lr * u).detach()

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.u = {}
        self.v = {}

    def step(self):
        self.t += 1
        for p in self.params:
            grad = p.grad + self.weight_decay * p
            u = self.u.get(p, 0.0)
            v = self.v.get(p, 0.0)
            u = self.beta1 * u + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad**2.0)
            self.u[p] = u.detach()
            self.v[p] = v.detach()
            u_hat = u / (1.0 - self.beta1**self.t)
            v_hat = v / (1.0 - self.beta2**self.t)
            p.data = (p - self.lr * u_hat / (v_hat**0.5 + self.eps)).detach()
