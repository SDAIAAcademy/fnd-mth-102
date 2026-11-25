"""Problem set 4 · Vector chain rules and the engine (week 4, 50 minutes, graded, auto-checked).

AI-off for the first attempt. Check with
    uv run python problem_sets/check.py PS4
from labs/. The checker compares with finite differences; it holds no answer key. Hand in this
file before the week 5 session. Replace each `raise NotImplementedError` with your answer.
"""
import math

import numpy as np


def vjp_matvec_x(A, g):
    """1 · y = A @ x, with A of shape (m, n). Given the upstream gradient g = dL/dy (length m),
    return dL/dx (length n). Do not build anything larger than A."""
    raise NotImplementedError


def vjp_matvec_A(x, g):
    """2 · Same y = A @ x. Return dL/dA, the same shape as A, given x (length n) and g (length m)."""
    raise NotImplementedError


def vjp_tanh(x, g):
    """3 · y = tanh(x) element by element. Return dL/dx given x and g = dL/dy (arrays of one length)."""
    raise NotImplementedError


def jac_composite(x):
    """4 · G(x) = (x0 * x1, x0 + x1) and F(u) = (sin(u0), u0 * u1, exp(u1)). Return the 3 x 2 Jacobian
    of F(G(x)) at the NumPy array x, as the product of the two Jacobians."""
    raise NotImplementedError


def make_sin(Value):
    """5 · Return a function sin(v) that takes an engine Value and returns a new Value with the right
    value and a _backward that adds cos(v.data) * out.grad to v.grad. Use the constructor
    Value(data, parents, op)."""
    raise NotImplementedError


# 6 · In two or three sentences: a layer maps 1,000 inputs to 1,000 outputs. Why does reverse mode
# pass a vector-Jacobian product backwards instead of building the layer's Jacobian?
why_vjp = "Write your answer here."

_ = (math, np)  # here for your answers
