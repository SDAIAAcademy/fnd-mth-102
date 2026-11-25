"""Problem set 5 · Gradient descent (week 5, 50 minutes, graded, auto-checked).

AI-off for the first attempt. Check with
    uv run python problem_sets/check.py PS5
from labs/. The checker tests behaviour on its own small problems; it holds no answer key. Hand
in this file before the week 6 session. Replace each `raise NotImplementedError` with your answer.
"""
import numpy as np


def gd(grad, w0, lr, steps):
    """1 · Plain gradient descent: start at w0 (a NumPy array), take `steps` steps of size lr against
    grad(w), and return the final w."""
    raise NotImplementedError


def largest_stable_rate():
    """2 · For L(w) = 0.5 * w @ H @ w with H = [[3, 1], [1, 3]], the learning rate above which plain
    gradient descent diverges. A number."""
    raise NotImplementedError


def momentum_step(w, v, g, lr, beta):
    """3 · One heavy-ball step: v_new = beta * v + g, w_new = w - lr * v_new. Return (w_new, v_new)."""
    raise NotImplementedError


def minibatch_grad(A, b, w, idx):
    """4 · The gradient of 0.5 * mean((A[idx] @ w - b[idx])**2) with respect to w."""
    raise NotImplementedError


def diagnose(hist, reference):
    """5 · Return "diverged", "slow", "noisy" or "converged" for a list of losses, with the rules of
    Lab 5.1 (10% above the reference is slow; a spread over the last 20 values above 5% of the
    reference is noisy)."""
    raise NotImplementedError


# 6 · In two or three sentences: why does stochastic gradient descent with a constant rate keep
# moving near the minimum, and what does a decaying rate change?
why_noise = "Write your answer here."

_ = np  # here for your answers
