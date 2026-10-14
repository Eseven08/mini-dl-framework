# -*- coding: utf-8 -*-
"""
Example 1 — Linear regression with minidl.
Learn the mapping y = 2x + 1 from noisy samples, using Linear + MSE + SGD.

Run:  python examples/linear_regression.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from minidl.loss import mse
from minidl.nn import Linear
from minidl.optim import SGD
from minidl.tensor import Tensor


def main():
    rng = np.random.default_rng(0)
    x = rng.uniform(-2.0, 2.0, (200, 1))
    y = 2.0 * x + 1.0 + rng.normal(0.0, 0.1, (200, 1))  # truth + noise

    model = Linear(1, 1, seed=0)
    optim = SGD(model.parameters(), lr=0.05)

    print("=" * 52)
    print("Linear regression: learn y = 2x + 1 from 200 noisy samples")
    print("=" * 52)

    for epoch in range(1, 501):
        optim.zero_grad()
        pred = model(Tensor(x))
        loss = mse(pred, Tensor(y))
        loss.backward()
        optim.step()
        if epoch % 100 == 0:
            print(f"epoch {epoch:>4d} | loss = {loss.data:.6f}")

    w = float(model.W.data[0, 0])
    b = float(model.b.data[0])
    print(f"\nlearned: y = {w:.3f} * x + {b:.3f}   (truth: y = 2.000 * x + 1.000)")


if __name__ == "__main__":
    main()
