# -*- coding: utf-8 -*-
"""
minidl.loss — loss functions.

`mse` is implemented purely through the autograd graph. `cross_entropy` uses a
manually-defined backward pass (softmax minus one-hot, scaled by batch size),
which is more numerically stable and cheaper than routing through the graph.
"""

import numpy as np

from .tensor import Tensor


def mse(pred, target):
    """Mean squared error. Both `pred` and `target` are Tensors."""
    diff = pred - target
    return (diff ** 2).mean()


def cross_entropy(logits, targets):
    """
    Softmax cross-entropy with a numerically stable manual backward.

    Parameters
    ----------
    logits : Tensor of shape (N, C) — raw, unnormalized class scores
    targets : 1D numpy array of class indices in [0, C)

    Returns a scalar Tensor (the mean loss over the batch).
    """
    l = logits.data
    n, c = l.shape

    # Stable softmax: subtract the row maximum before exponentiating,
    # otherwise exp() can overflow.
    m = l.max(axis=1, keepdims=True)
    e = np.exp(l - m)
    probs = e / e.sum(axis=1, keepdims=True)

    picked = probs[np.arange(n), targets]
    loss = float(-np.log(np.maximum(picked, 1e-12)).mean())

    out = Tensor(loss, requires_grad=logits.requires_grad)
    if logits.requires_grad:
        def _backward():
            # dL/dz = softmax(z) - one_hot(target), averaged over the batch
            d = probs.copy()
            d[np.arange(n), targets] -= 1.0
            logits.grad += d / n
        out._backward = _backward
        out._prev = (logits,)
    return out
