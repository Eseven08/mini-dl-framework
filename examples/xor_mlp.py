# -*- coding: utf-8 -*-
"""
Example 2 — Solve XOR with a two-layer MLP built on minidl.
The classic "linearly inseparable" problem, solved by adding a hidden layer.

Run:  python examples/xor_mlp.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from minidl.loss import mse
from minidl.nn import Linear, Sequential, tanh
from minidl.optim import SGD
from minidl.tensor import Tensor

DATA = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]

# Batch matrices: MSE over the whole batch averages the gradients, which keeps
# the update scale consistent no matter how large the batch is.
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
Y = np.array([[0], [1], [1], [0]], dtype=np.float64)


def main():
    model = Sequential(Linear(2, 8, seed=1), tanh, Linear(8, 1, seed=1))
    # lr=0.5 diverges for this architecture (a nice experiment to try!);
    # 0.2 converges reliably to near-zero loss.
    optim = SGD(model.parameters(), lr=0.2)

    print("=" * 50)
    print("Training a 2-8-1 MLP (minidl) to learn XOR")
    print("=" * 50)

    for epoch in range(1, 10001):
        optim.zero_grad()
        pred = model(Tensor(X))
        loss = mse(pred, Tensor(Y))
        loss.backward()
        optim.step()
        if epoch % 2000 == 0 or epoch == 1:
            print(f"epoch {epoch:>6d} | loss = {loss.data:.6f}")

    print("\nTest results (threshold 0.5):")
    print("-" * 50)
    ok = 0
    for x, target in DATA:
        out = float(model(Tensor(np.array(x, dtype=np.float64))).data[0])
        pred = 1 if out >= 0.5 else 0
        flag = "OK" if pred == target else "FAIL"
        ok += pred == target
        print(f"  input {x} -> expected {target}, got {out:.4f} -> {pred}  {flag}")
    print("-" * 50)
    print(f"accuracy: {ok}/{len(DATA)}")


if __name__ == "__main__":
    main()
