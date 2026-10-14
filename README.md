# mini-dl-framework

A **minimal deep learning framework with automatic differentiation**, written from scratch in Python on top of NumPy. It is a mini "PyTorch from first principles": a `Tensor` that records a computation graph, reverse-mode autograd, layers, optimizers and losses — enough to train real neural networks, with the entire engine visible in ~400 lines of readable code.

The goal of this project is **depth, not breadth**: by reading every line, you understand exactly how modern deep learning frameworks work under the hood — automatic differentiation, backpropagation, optimizers and the training loop.

## Features

- **Autograd engine** (`minidl/tensor.py`)
  - `Tensor` with `.data` and `.grad`
  - Reverse-mode automatic differentiation over an explicit computation graph (topological-order backward pass)
  - Ops: add, sub, mul, div, neg, matmul, pow, exp, log, relu, sigmoid, tanh, softmax, mean, sum, reshape — with correct broadcasting and accumulation of gradients
- **Layers** (`minidl/nn.py`): `Linear`, `Sequential`, activation functions, trainable `Parameter`
- **Optimizers** (`minidl/optim.py`): `SGD` (with momentum), `Adam` (with bias correction)
- **Losses** (`minidl/loss.py`): `MSE`, `CrossEntropy` (numerically stable, manual backward)
- **Verified correctness**: gradient-checking tests compare analytical gradients against numerical (central-difference) gradients for every op

## Requirements

- Python 3.9+
- NumPy (`pip install numpy`)

## Quick Start

```bash
# Run the gradient-checking tests (prove the autograd engine is correct)
python tests/test_gradient.py

# Run the end-to-end model tests
python tests/test_models.py

# Run the examples
python examples/linear_regression.py
python examples/xor_mlp.py
python examples/spiral_classification.py
```

## Examples

| Example | Problem | What it shows |
| --- | --- | --- |
| `linear_regression.py` | learn `y = 2x + 1` from noisy data | Linear + MSE + SGD |
| `xor_mlp.py` | solve XOR (linearly inseparable) | a hidden layer gives nonlinearity |
| `spiral_classification.py` | separate two intertwined spirals | deeper MLP + ReLU + CrossEntropy + Adam learning a nonlinear decision boundary (~95%+ accuracy) |

## How the Autograd Engine Works

Consider `loss = ((x @ W + b) - y) ** 2`:

```
        x ─┐
        W ─┼─► (matmul) ─► (+b) ─► (sub) ─► (pow 2) ─► (mean) ─► loss
             │                │
             └──────── b ─────┘        y ─┘
```

Every op produces an output `Tensor` and stores a closure `_backward` that knows how to push gradients into its parents. Calling `loss.backward()`:

1. Builds a **topological ordering** of the graph (parents before children).
2. Walks the ordering **in reverse** (children first).
3. At each node, runs its `_backward`, which adds `∂loss/∂output × local ∂output/∂input` into each parent's `.grad` — this *is* the chain rule.

Gradients accumulate (`+=`), which is what makes batch training work: multiple forward/backward passes before one optimizer step sum up naturally.

## Project Structure

```
mini-dl-framework/
├── minidl/                  # the framework itself
│   ├── tensor.py            # Tensor + reverse-mode autograd engine
│   ├── nn.py                # Linear, Sequential, activations
│   ├── optim.py             # SGD, Adam
│   └── loss.py              # MSE, CrossEntropy
├── examples/                # runnable demos
│   ├── linear_regression.py
│   ├── xor_mlp.py
│   └── spiral_classification.py
├── tests/
│   ├── test_gradient.py     # numerical gradient checking for every op
│   └── test_models.py       # end-to-end training tests
├── README.md
└── requirements.txt
```

## Why "from scratch"?

Most ML learners call `loss.backward()` in PyTorch every day without knowing what happens inside. This framework removes the magic:

- You can **read the entire backward pass** in `tensor.py` — there is no C++ engine hiding anything.
- You can **prove it correct** yourself: `tests/test_gradient.py` verifies every gradient numerically.
- You can **extend it**: add `Conv2d`, `Dropout`, `BatchNorm`, or a `DataLoader` — the graph-based design makes new ops a 10-line job.

## Ideas to Extend

- Add a `Conv2d` layer + `max_pool` and train on MNIST
- Add `Dropout` and `BatchNorm1d`
- Add a `DataLoader` with shuffling and mini-batches
- Implement `SoftmaxCrossEntropy` as a graph-based op and compare speed with the manual one
- Add a `nn.MSELoss`-style class API (callable loss objects)
- Benchmark your framework vs PyTorch on the same MLP

## License

MIT License — see [LICENSE](LICENSE).
