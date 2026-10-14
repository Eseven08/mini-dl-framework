# -*- coding: utf-8 -*-
"""
minidl — a minimal deep learning framework with automatic differentiation.

Package layout:
    minidl.tensor   Tensor + reverse-mode autograd engine
    minidl.nn       layers (Linear, Sequential) and activations
    minidl.optim    optimizers (SGD, Adam)
    minidl.loss     loss functions (MSE, CrossEntropy)
"""

from .tensor import Tensor
from . import nn, optim, loss

__version__ = "0.1.0"
__all__ = ["Tensor", "nn", "optim", "loss"]
