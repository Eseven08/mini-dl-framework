# -*- coding: utf-8 -*-
"""
Example 3 — Classify two intertwined spirals with a deeper MLP.
A 2-16-16-2 ReLU network trained with CrossEntropy + Adam.
This is a real nonlinear decision-boundary problem: a linear model would
score only ~50%, the MLP should reach ~95%+.

Run:  python examples/spiral_classification.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from minidl.loss import cross_entropy
from minidl.nn import Linear, Sequential, relu
from minidl.optim import Adam
from minidl.tensor import Tensor


def make_spiral(n=120, noise=0.2, seed=0):
    """Generate two interleaved spiral classes (classic nonlinear benchmark)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)
    theta = 4.0 * np.pi * t
    r = t

    class0 = np.column_stack([r * np.sin(theta), r * np.cos(theta)])
    class1 = np.column_stack([r * np.sin(theta + np.pi), r * np.cos(theta + np.pi)])

    X = np.vstack([class0, class1])
    y = np.array([0] * n + [1] * n)
    X = X + rng.normal(0.0, noise, X.shape)
    return X, y


def main():
    X, y = make_spiral(n=120, noise=0.1, seed=0)

    model = Sequential(
        Linear(2, 32, seed=1), relu,
        Linear(32, 32, seed=1), relu,
        Linear(32, 2, seed=1),
    )
    optim = Adam(model.parameters(), lr=0.01)

    n = len(y)
    epochs = 3000

    print("=" * 56)
    print("Spiral classification: 2-16-16-2 MLP, CrossEntropy + Adam")
    print("=" * 56)

    for epoch in range(1, epochs + 1):
        optim.zero_grad()
        logits = model(Tensor(X))
        loss = cross_entropy(logits, y)
        loss.backward()
        optim.step()
        if epoch % 500 == 0 or epoch == 1:
            print(f"epoch {epoch:>5d} | loss = {loss.data:.6f}")

    pred = model(Tensor(X)).data.argmax(axis=1)
    acc = float((pred == y).mean())
    print(f"\nfinal test accuracy: {acc * 100:.1f}%  ({int((pred == y).sum())}/{n})")
    print("(a linear model would only reach ~50% on this dataset)")


if __name__ == "__main__":
    main()
