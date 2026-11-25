"""Problem set 3 · The chain rule on graphs (week 3, 50 minutes, graded, auto-checked).

AI-off for the first attempt: draw each graph and run each backward pass on paper, then code
your answers here. Check with
    uv run python problem_sets/check.py PS3
from labs/. The checker compares with finite differences; it holds no answer key. Hand in this
file before the week 4 session. Replace each `raise NotImplementedError` with your answer.
"""
import math

import numpy as np


def grad_q(x, y):
    """1 · The gradient of q(x, y) = (x * y + y)**2 as a NumPy array [dq/dx, dq/dy]. y is used twice: sum both paths."""
    raise NotImplementedError


def grad_g(a, b, c):
    """2 · The gradient of g(a, b, c) = (a + b) * c + exp(a), as a NumPy array [dg/da, dg/db, dg/dc]."""
    raise NotImplementedError


def logistic_grad(w, b):
    """3 · One learner with x = [0.3, -1.2, 2.0] and y = 0. Loss l = -(y log p + (1 - y) log(1 - p)),
    p = sigmoid(w @ x + b). Return (dl/dw as a NumPy array of length 3, dl/db as a float)."""
    raise NotImplementedError


def z_adjoint(p, y):
    """4 · For one learner, the derivative of the cross-entropy loss with respect to z, written with p = sigmoid(z) and y."""
    raise NotImplementedError


def passes(n_inputs, n_outputs):
    """5 · The number of passes each mode needs for the full Jacobian of a function with n_inputs inputs and
    n_outputs outputs, as a dict {"forward": ..., "backward": ...}."""
    raise NotImplementedError


# 6 · In two or three sentences: in the graph of question 1, why is the adjoint of y a sum? What
# would a backward pass that overwrites, instead of adding, get wrong?
why_sum = "Write your answer here."

_ = (math, np)  # here for your answers
