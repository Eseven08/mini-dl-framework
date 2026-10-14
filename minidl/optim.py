# -*- coding: utf-8 -*-
"""
minidl.optim — optimizers: SGD (with optional momentum) and Adam.

An optimizer owns a list of Parameters. Usage pattern:
    optim.zero_grad()   # clear gradients from the previous step
    loss.backward()     # accumulate gradients
    optim.step()        # update the parameters
"""

import numpy as np


class SGD:
    """Stochastic gradient descent with optional momentum."""

    def __init__(self, params, lr=0.01, momentum=0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad.fill(0.0)

    def step(self):
        for i, p in enumerate(self.params):
            # v = m*v - lr*grad ;  p += v   (momentum=0 -> plain SGD)
            self.velocity[i] = (self.momentum * self.velocity[i]
                                - self.lr * p.grad)
            p.data += self.velocity[i]


class Adam:
    """Adam optimizer with bias correction (Kingma & Ba, 2015)."""

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.params]  # first moment
        self.v = [np.zeros_like(p.data) for p in self.params]  # second moment
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad.fill(0.0)

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g * g)
            # Bias correction: moments start at zero, so early steps are scaled up
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
