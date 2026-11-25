"""Problem set 2 · Gradients and Jacobians (week 2, 50 minutes, graded, auto-checked).

AI-off for the first attempt: work on paper, then code your answers here. Check with
    uv run python problem_sets/check.py PS2
from labs/. The checker compares with finite differences; it holds no answer key. Hand in this
file before the week 3 session. Replace each `raise NotImplementedError` with your answer.
"""
import math

import numpy as np


def grad_f(x, y):
    """1 · The gradient of f(x, y) = x**2 * y + 3 * y**3, as a NumPy array [df/dx, df/dy]."""
    raise NotImplementedError


def grad_g(w):
    """2 · The gradient of g(w) = log(1 + exp(w @ z)) with z = [1, -2, 0.5], for a NumPy array w of length 3."""
    raise NotImplementedError


def directional():
    """3 · The directional derivative of f (question 1) at (1, 2) along the unit vector (3/5, 4/5). A number."""
    raise NotImplementedError


def jac_F(x, y):
    """4 · The 3 x 2 Jacobian of F(x, y) = (x * y, x + y**2, sin(x)), as a NumPy array."""
    raise NotImplementedError


def steepest_descent_direction():
    """5 · The unit vector along which f (question 1) decreases fastest at (1, 2), as a NumPy array."""
    raise NotImplementedError


# 6 · In two or three sentences: why does a gradient checker compare a relative error, not the
# absolute difference, with its tolerance? Give an example where the absolute difference misleads.
why_relative = "Write your answer here."

_ = (math, np)  # here for your answers
