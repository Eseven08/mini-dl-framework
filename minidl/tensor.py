# -*- coding: utf-8 -*-
"""
minidl.tensor — Tensor with reverse-mode automatic differentiation (autograd)

This module is the heart of mini-dl-framework. A `Tensor` wraps a numpy array
and records the operations that created it. Calling `.backward()` on a scalar
loss walks the computation graph in reverse (topological order) and, for every
node, runs its saved `_backward` closure, which accumulates gradients into its
parents. This is exactly how PyTorch's autograd works, minus the C++ engine.

Key ideas:
  * Every tensor has `.data` (its value) and `.grad` (accumulated gradient).
  * An op creates an output Tensor and stores a closure `_backward` that
    computes the local gradient and adds it into each parent's `.grad`.
  * The chain rule is applied automatically by the reverse topological walk.

Run the gradient-checking tests to see the engine verified numerically:
    python tests/test_gradient.py
"""

import numpy as np


def _reduce_grad(grad, shape):
    """Sum `grad` back to `shape`, undoing numpy broadcasting.

    If we computed `out = a + b` with broadcasting, d(out)/d(a) must have the
    same shape as `a`. Gradients flow through broadcast dimensions by summing
    over the extra / repeated axes.
    """
    g = grad
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i in range(len(shape)):
        if shape[i] == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


class Tensor:
    """A numpy-backed tensor that participates in a computation graph."""

    def __init__(self, data, requires_grad=False):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._backward = None  # closure invoked during backward()
        self._prev = ()        # parent tensors in the computation graph

    # ------------------------------------------------------------------ #
    # Small helpers
    # ------------------------------------------------------------------ #
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"

    # ------------------------------------------------------------------ #
    # Autograd
    # ------------------------------------------------------------------ #
    def backward(self):
        """Run reverse-mode autodiff from this tensor (assumed scalar loss)."""
        self.grad = np.ones_like(self.data)  # d(loss)/d(loss) = 1

        # Build a topological ordering: every parent comes before its children.
        topo = []
        visited = set()

        def build(v):
            if id(v) in visited:
                return
            visited.add(id(v))
            for parent in v._prev:
                build(parent)
            topo.append(v)

        build(self)

        # Walk in reverse (children first). Each node pushes its local gradient
        # into its parents; by the time a node is processed, all downstream
        # contributions have already been accumulated into its `.grad`.
        for v in reversed(topo):
            if v._backward is not None:
                v._backward()

    # ------------------------------------------------------------------ #
    # Binary operations
    # ------------------------------------------------------------------ #
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data,
                     requires_grad=self.requires_grad or other.requires_grad)
        out._prev = (self, other)

        def _backward():
            if self.requires_grad:
                self.grad += _reduce_grad(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _reduce_grad(out.grad, other.data.shape)
        out._backward = _backward
        return out

    __radd__ = __add__

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data,
                     requires_grad=self.requires_grad or other.requires_grad)
        out._prev = (self, other)

        def _backward():
            if self.requires_grad:
                self.grad += _reduce_grad(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad -= _reduce_grad(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __rsub__(self, other):
        return Tensor(other) - self

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,
                     requires_grad=self.requires_grad or other.requires_grad)
        out._prev = (self, other)

        def _backward():
            if self.requires_grad:
                self.grad += _reduce_grad(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += _reduce_grad(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data,
                     requires_grad=self.requires_grad or other.requires_grad)
        out._prev = (self, other)

        def _backward():
            if self.requires_grad:
                self.grad += _reduce_grad(out.grad / other.data, self.data.shape)
            if other.requires_grad:
                other.grad -= _reduce_grad(out.grad * self.data / (other.data ** 2),
                                           other.data.shape)
        out._backward = _backward
        return out

    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad)
        out._prev = (self,)

        def _backward():
            self.grad += -out.grad
        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        a, b = self.data, other.data
        a_was_1d = (a.ndim == 1)
        a2 = a.reshape(1, -1) if a_was_1d else a
        out_data = a2 @ b
        if a_was_1d:
            out_data = out_data.reshape(-1)

        out = Tensor(out_data,
                     requires_grad=self.requires_grad or other.requires_grad)
        out._prev = (self, other)

        def _backward():
            grad2 = out.grad.reshape(1, -1) if a_was_1d else out.grad
            if self.requires_grad:
                ga = grad2 @ b.T
                if a_was_1d:
                    ga = ga.reshape(-1)
                self.grad += ga
            if other.requires_grad:
                other.grad += a2.T @ grad2
        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Reductions
    # ------------------------------------------------------------------ #
    def sum(self):
        out = Tensor(self.data.sum(), requires_grad=self.requires_grad)
        out._prev = (self,)

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out

    def mean(self):
        out = Tensor(self.data.mean(), requires_grad=self.requires_grad)
        out._prev = (self,)

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad / self.data.size
        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Unary math operations
    # ------------------------------------------------------------------ #
    def pow(self, p):
        out = Tensor(self.data ** p, requires_grad=self.requires_grad)
        out._prev = (self,)

        def _backward():
            self.grad += out.grad * p * (self.data ** (p - 1.0))
        out._backward = _backward
        return out

    __pow__ = pow

    def exp(self):
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad)
        out._prev = (self,)

        def _backward():
            self.grad += out.grad * out.data  # d(exp(x))/dx = exp(x)
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad)
        out._prev = (self,)

        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), requires_grad=self.requires_grad)
        out._prev = (self,)

        def _backward():
            self.grad += out.grad * (self.data > 0.0)
        out._backward = _backward
        return out

    def sigmoid(self):
        x = self.data
        # Numerically stable sigmoid (no overflow for large |x|)
        y = np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                     np.exp(x) / (1.0 + np.exp(x)))
        out = Tensor(y, requires_grad=self.requires_grad)
        out._prev = (self,)

        def _backward():
            self.grad += out.grad * y * (1.0 - y)
        out._backward = _backward
        return out

    def tanh(self):
        y = np.tanh(self.data)
        out = Tensor(y, requires_grad=self.requires_grad)
        out._prev = (self,)

        def _backward():
            self.grad += out.grad * (1.0 - y * y)
        out._backward = _backward
        return out

    def softmax(self):
        """Row-wise softmax (last axis)."""
        x = self.data
        m = x.max(axis=-1, keepdims=True)
        e = np.exp(x - m)
        p = e / e.sum(axis=-1, keepdims=True)
        out = Tensor(p, requires_grad=self.requires_grad)
        out._prev = (self,)

        def _backward():
            # dL/dx = p * (grad - sum(grad * p, axis=-1))
            s = (out.grad * p).sum(axis=-1, keepdims=True)
            self.grad += p * (out.grad - s)
        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad)
        out._prev = (self,)

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out
