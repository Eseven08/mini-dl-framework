# -*- coding: utf-8 -*-
"""
End-to-end model tests — each test trains a real model with minidl and
asserts that it reaches the expected accuracy. These prove that the whole
pipeline (autograd -> layers -> optimizer -> loss) works together.

Run:  python tests/test_models.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from minidl.loss import cross_entropy, mse
from minidl.nn import Linear, Sequential, relu, tanh
from minidl.optim import Adam, SGD
from minidl.tensor import Tensor


def test_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    Y = np.array([[0], [1], [1], [0]], dtype=np.float64)
    data = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
    model = Sequential(Linear(2, 8, seed=1), tanh, Linear(8, 1, seed=1))
    optim = SGD(model.parameters(), lr=0.2)

    for _ in range(5000):
        optim.zero_grad()
        mse(model(Tensor(X)), Tensor(Y)).backward()
        optim.step()

    correct = 0
    for x, target in data:
        out = float(model(Tensor(np.array(x, dtype=np.float64))).data[0])
        correct += (1 if out >= 0.5 else 0) == target
    assert correct == 4, f"XOR: got {correct}/4 correct"
    print(f"[PASS] XOR solved by minidl MLP: {correct}/4")


def test_linear_regression():
    rng = np.random.default_rng(0)
    x = rng.uniform(-2.0, 2.0, (200, 1))
    y = 2.0 * x + 1.0 + rng.normal(0.0, 0.1, (200, 1))

    model = Linear(1, 1, seed=0)
    optim = SGD(model.parameters(), lr=0.05)
    for _ in range(1500):
        optim.zero_grad()
        mse(model(Tensor(x)), Tensor(y)).backward()
        optim.step()

    w = float(model.W.data[0, 0])
    b = float(model.b.data[0])
    assert abs(w - 2.0) < 0.1 and abs(b - 1.0) < 0.1, f"regression: w={w:.3f} b={b:.3f}"
    print(f"[PASS] Linear regression: y = {w:.3f}x + {b:.3f} (truth 2.000, 1.000)")


def test_spiral():
    rng = np.random.default_rng(0)
    n = 120
    t = np.linspace(0.0, 1.0, n)
    theta = 4.0 * np.pi * t
    r = t
    c0 = np.column_stack([r * np.sin(theta), r * np.cos(theta)])
    c1 = np.column_stack([r * np.sin(theta + np.pi), r * np.cos(theta + np.pi)])
    X = np.vstack([c0, c1]) + rng.normal(0.0, 0.1, (2 * n, 2))
    y = np.array([0] * n + [1] * n)

    model = Sequential(Linear(2, 32, seed=1), relu,
                       Linear(32, 32, seed=1), relu,
                       Linear(32, 2, seed=1))
    optim = Adam(model.parameters(), lr=0.01)
    for _ in range(2000):
        optim.zero_grad()
        cross_entropy(model(Tensor(X)), y).backward()
        optim.step()

    pred = model(Tensor(X)).data.argmax(axis=1)
    acc = float((pred == y).mean())
    assert acc > 0.9, f"spiral accuracy too low: {acc:.3f}"
    print(f"[PASS] Spiral classification accuracy: {acc * 100:.1f}%")


def main():
    print("=" * 56)
    print("End-to-end model tests (minidl)")
    print("=" * 56)
    test_xor()
    test_linear_regression()
    test_spiral()
    print("\nAll model tests passed.")


if __name__ == "__main__":
    main()
