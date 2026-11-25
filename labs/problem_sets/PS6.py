"""Problem set 6 · Convexity and curvature (week 6, 50 minutes, graded, auto-checked).

AI-off for the first attempt. Check with
    uv run python problem_sets/check.py PS6
from labs/. The checker uses finite differences and its own runs; it holds no answer key. Hand in
this file before the week 7 session. Replace each `raise NotImplementedError` with your answer.
"""
import numpy as np


def hessian_f(x, y):
    """1 · The Hessian of f(x, y) = x**3 + 2*x*y + y**2, as a 2 x 2 NumPy array."""
    raise NotImplementedError


def kind_at(x, y):
    """2 · g(x, y) = x**3 - 3*x + y**2 has critical points (1, 0) and (-1, 0). Return "minimum", "maximum"
    or "saddle" for the point (x, y) given."""
    raise NotImplementedError


def stable_step():
    """3 · For L(w) = 0.5 * w @ H @ w with H = [[4, 1], [1, 2]], the step size above which gradient descent
    diverges. A number (an expression with np.sqrt is fine)."""
    raise NotImplementedError


def convex_flags():
    """4 · For each function, True if it is convex on the whole plane: {"a": ..., "b": ..., "c": ...}
    a: x**4 + y**2     b: x**2 - y**2 + x*y     c: log(1 + exp(x)) + (x - y)**2"""
    raise NotImplementedError


def newton_step(H, g):
    """5 · The Newton step d that solves H d = -g, for a 2 x 2 NumPy array H and a gradient g."""
    raise NotImplementedError


# 6 · In two or three sentences: why can gradient descent spend a long time near a saddle point,
# and why does a little noise help?
why_saddle = "Write your answer here."

_ = np  # here for your answers
