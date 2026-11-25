"""Problem set 1 · Derivatives and approximation (week 1, 50 minutes, graded, auto-checked).

AI-off for the first attempt: work on paper, then code your answers here. Check with
    uv run python problem_sets/check.py PS1
from labs/. The checker compares with finite differences; it holds no answer key. Hand in this
file before the week 2 session. Replace each `raise NotImplementedError` with your answer.
"""
import math


def d1(x):
    """1 · The derivative of f(x) = x**3 * exp(-x). Use the product rule."""
    raise NotImplementedError


def d2(x):
    """2 · The derivative of g(x) = log(1 + x**2) / (1 + x). Use the quotient (or product) rule and the chain rule."""
    raise NotImplementedError


def tangent_at_3():
    """3 · The tangent line of h(x) = sqrt(1 + x) at x = 3, as (slope, intercept) of y = slope * x + intercept."""
    raise NotImplementedError


def taylor_terms():
    """4 · For k(x) = exp(-x**2) at x = 0.5: (k(0.5), k'(0.5), k''(0.5)), the three numbers of the quadratic Taylor model."""
    raise NotImplementedError


def best_step():
    """5 · Among h = 1e-1, 1e-2, ..., 1e-12, the step whose central difference of exp at x = 2 has the
    smallest absolute error. Return it as a float, such as 1e-4. Compute it; do not guess."""
    raise NotImplementedError


# 6 · In two or three sentences: why does a finite-difference step of 1e-12 usually give a worse
# derivative than a step of 1e-6? Name both kinds of error.
why_not_smaller = "Write your answer here."

_ = math  # math is here for your answers
