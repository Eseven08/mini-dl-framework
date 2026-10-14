# -*- coding: utf-8 -*-
"""
minidl.nn — neural network building blocks built on the autograd Tensor.

Layers hold their weights as `Parameter` objects (Tensors with
requires_grad=True), so an optimizer can step them after a backward pass.
"""

import math

import numpy as np

from .tensor import Tensor


class Parameter(Tensor):
    """A Tensor that an optimizer is allowed to train (requires_grad=True)."""

    def __init__(self, data):
        super().__init__(np.asarray(data, dtype=np.float64), requires_grad=True)


class Linear:
    """Fully-connected layer: y = x @ W + b, with W of shape (in_features, out_features)."""

    def __init__(self, in_features, out_features, seed=None):
        rng = np.random.default_rng(seed)
        # Uniform init scaled by 1/sqrt(in), keeps activations well-behaved
        k = 1.0 / math.sqrt(in_features)
        self.W = Parameter(rng.uniform(-k, k, (in_features, out_features)))
        self.b = Parameter(np.zeros(out_features))

    def __call__(self, x):
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]


class Sequential:
    """Stack layers / functions and apply them in order."""

    def __init__(self, *layers):
        self.layers = list(layers)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
        return params


# Activation functions: thin wrappers over the Tensor methods
def relu(x):
    return x.relu()


def sigmoid(x):
    return x.sigmoid()


def tanh(x):
    return x.tanh()


def softmax(x):
    return x.softmax()
