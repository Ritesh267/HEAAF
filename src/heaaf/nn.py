"""A small dependency-free MLP.

Written directly in NumPy for three reasons: the whole pipeline then runs on a
single CPU core with no deep-learning framework, the results are bit-for-bit
reproducible from a seed, and the backward pass with respect to the *inputs*
(needed for gradient-based attribution) is available without any autograd
plumbing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


def he_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))


@dataclass
class Adam:
    lr: float = 1e-3
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8

    def __post_init__(self):
        self.m: List[np.ndarray] = []
        self.v: List[np.ndarray] = []
        self.t = 0

    def step(self, params: List[np.ndarray], grads: List[np.ndarray]) -> None:
        if not self.m:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            p -= self.lr * (self.m[i] / bc1) / (np.sqrt(self.v[i] / bc2) + self.eps)


class MLP:
    """ReLU multilayer perceptron with a linear output head."""

    def __init__(self, sizes: Sequence[int], seed: int = 0, lr: float = 1e-3):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.W = [he_init(a, b, rng) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.opt = Adam(lr=lr)

    # -- parameters ----------------------------------------------------
    @property
    def params(self) -> List[np.ndarray]:
        return self.W + self.b

    def copy_from(self, other: "MLP") -> None:
        for i in range(len(self.W)):
            self.W[i][...] = other.W[i]
            self.b[i][...] = other.b[i]

    def clone(self) -> "MLP":
        c = MLP(self.sizes, seed=0)
        c.copy_from(self)
        return c

    # -- forward -------------------------------------------------------
    def forward(self, X: np.ndarray, cache: bool = False):
        A = np.atleast_2d(X)
        acts = [A]
        pre = []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            Z = A @ W + b
            pre.append(Z)
            A = np.maximum(Z, 0.0) if i < len(self.W) - 1 else Z
            acts.append(A)
        if cache:
            return A, acts, pre
        return A

    __call__ = forward

    # -- backward w.r.t. parameters ------------------------------------
    def backward(self, acts, pre, dY: np.ndarray, grad_clip: float | None = None):
        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.b)
        delta = dY
        for i in reversed(range(len(self.W))):
            grads_W[i] = acts[i].T @ delta / delta.shape[0]
            grads_b[i] = delta.mean(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * (pre[i - 1] > 0)
        if grad_clip:
            nrm = np.sqrt(sum((g ** 2).sum() for g in grads_W + grads_b))
            if nrm > grad_clip:
                sc = grad_clip / (nrm + 1e-12)
                grads_W = [g * sc for g in grads_W]
                grads_b = [g * sc for g in grads_b]
        return grads_W, grads_b

    def apply_grads(self, grads_W, grads_b) -> None:
        self.opt.step(self.W + self.b, list(grads_W) + list(grads_b))

    # -- backward w.r.t. inputs (used by attribution) -------------------
    def input_gradient(self, X: np.ndarray, head: np.ndarray) -> np.ndarray:
        """d(head . output)/dX for a batch of inputs.

        ``head`` is a (n_out,) vector selecting the scalar functional whose
        gradient is required -- for HEAAF this is the risk head.
        """
        X = np.atleast_2d(X)
        _, acts, pre = self.forward(X, cache=True)
        delta = np.tile(head.reshape(1, -1), (X.shape[0], 1))
        for i in reversed(range(len(self.W))):
            if i > 0:
                delta = (delta @ self.W[i].T) * (pre[i - 1] > 0)
            else:
                delta = delta @ self.W[0].T
        return delta
