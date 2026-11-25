"""Problem set 7 · Stable losses and penalties (week 7, 50 minutes, graded, auto-checked).

AI-off for the first attempt. Check with
    uv run python problem_sets/check.py PS7
from labs/. The checker uses finite differences and its own runs; it holds no answer key. Hand in
this file before the week 8 session. Replace each `raise NotImplementedError` with your answer.
"""
import numpy as np


def log_sigmoid(z):
    """1 · log(sigmoid(z)) for a NumPy array z, finite for every z from -1000 to 1000."""
    raise NotImplementedError


def bce_from_logits(z, y):
    """2 · The mean binary cross-entropy of labels y (0 or 1) given logits z (arrays of the same length),
    computed from the logits without forming a probability; finite for logits of size 1000."""
    raise NotImplementedError


def ce_grad_row(z, y):
    """3 · The gradient of logsumexp(z) - z[y] with respect to the logits z (a 1-D array), for label y."""
    raise NotImplementedError


def ridge_grad(w, X, y, lam):
    """4 · The gradient with respect to w of 0.5 * mean((X @ w - y)**2) + 0.5 * lam * (w @ w)."""
    raise NotImplementedError


def ridge_solution(X, y, lam):
    """5 · The w that minimises the loss of question 4, for lam > 0 (solve a linear system)."""
    raise NotImplementedError


# 6 · In two or three sentences: on data a linear classifier can separate perfectly, why do the weights
# keep growing under gradient descent, and what does an L2 penalty change?
why_penalty = "Write your answer here."

_ = np  # here for your answers
