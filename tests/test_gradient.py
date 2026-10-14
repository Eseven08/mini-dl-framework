# -*- coding: utf-8 -*-
"""
Gradient checking — numerically verifies that the autograd engine computes
correct gradients for every operation.

Method: for each op we compare the analytical gradient (from .backward())
against a central-difference numerical gradient. If they agree to within a
small relative error, the chain rule implementation is correct.

Run:  python tests/test_gradient.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from minidl.loss import cross_entropy
from minidl.tensor import Tensor


def numerical_grad(f, x, eps=1e-5):
    """Central-difference approximation of df/dx, element by element."""
    g = np.zeros_like(x)
    for idx in np.ndindex(*x.shape):
        xp = x.copy()
        xm = x.copy()
        xp[idx] += eps
        xm[idx] -= eps
        g[idx] = (f(xp) - f(xm)) / (2.0 * eps)
    return g


def check(name, fn, x0, tol=1e-4):
    """fn: Tensor -> scalar Tensor. x0: numpy array. Compares grads."""
    xt = Tensor(x0, requires_grad=True)
    fn(xt).backward()

    ng = numerical_grad(lambda v: float(fn(Tensor(v)).data), x0)
    rel = np.abs(xt.grad - ng) / (np.abs(ng) + 1e-8)
    err = float(rel.max())
    status = "PASS" if err < tol else "FAIL"
    print(f"[{status}] {name:<24s} max relative error = {err:.2e}")
    assert err < tol, f"{name} failed with max relative error {err}"


def check_parameters():
    """Verify gradients of real trainable parameters through an MLP block."""
    rng = np.random.default_rng(2)
    X = Tensor(rng.normal(size=(8, 4)))
    W = Tensor(rng.normal(size=(4, 6)), requires_grad=True)
    b = Tensor(rng.normal(size=6), requires_grad=True)
    W1 = Tensor(rng.normal(size=(6, 2)), requires_grad=True)
    targets = rng.integers(0, 2, size=8)

    def loss_fn():
        h = (X @ W + b).relu()
        return cross_entropy(h @ W1, targets)

    loss_fn().backward()

    def fw(w):
        W.data = w
        return float(loss_fn().data)

    def fb(bv):
        b.data = bv
        return float(loss_fn().data)

    def fw1(w1):
        W1.data = w1
        return float(loss_fn().data)

    for name, param, f in [("W", W, fw), ("b", b, fb), ("W1", W1, fw1)]:
        ng = numerical_grad(f, param.data.copy())
        rel = np.abs(param.grad - ng) / (np.abs(ng) + 1e-8)
        err = float(rel.max())
        status = "PASS" if err < 1e-4 else "FAIL"
        print(f"[{status}] param {name:<24s} max relative error = {err:.2e}")
        assert err < 1e-4, f"param {name} failed with {err}"


def main():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 3))     # 2D input
    v = rng.normal(size=3)          # 1D input
    B = rng.normal(size=(3, 4))     # matmul weight
    w = rng.normal(size=(5, 3))     # weights for the softmax weighted sum

    print("=" * 66)
    print("Gradient checking: analytical (backward) vs numerical (central difference)")
    print("=" * 66)

    check("add", lambda x: (x + 1.7).sum(), A)
    check("sub", lambda x: (x - 1.1).sum(), A)
    check("mul", lambda x: (x * 2.3).sum(), A)
    check("div", lambda x: (x / 4.2).sum(), A)
    check("pow", lambda x: (x ** 3).sum(), A)
    check("matmul (2D)", lambda x: (x @ B).sum(), A)
    check("matmul (1D)", lambda x: (x @ B).sum(), v)
    check("exp", lambda x: x.exp().sum(), A)
    check("log", lambda x: (x + 5.0).log().sum(), A)
    check("relu", lambda x: x.relu().sum(), A)
    check("sigmoid", lambda x: x.sigmoid().sum(), A)
    check("tanh", lambda x: x.tanh().sum(), A)
    check("softmax (weighted)", lambda x: (x.softmax() * Tensor(w)).sum(), A)
    check("mean", lambda x: (x + 1.0).mean(), A)

    print()
    check_parameters()
    print("\nAll gradient checks passed — the autograd engine is correct.")


if __name__ == "__main__":
    main()
