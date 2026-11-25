"""Problem set 8 · Constraints and prices (week 8, 50 minutes, graded, auto-checked).

AI-off for the first attempt. Check with
    uv run python problem_sets/check.py PS8
from labs/. The checker tests the conditions your answers must satisfy and runs its own searches;
it holds no answer key. Hand in this file before Test 2. Replace each `raise NotImplementedError`
with your answer.
"""
import numpy as np


def area_problem():
    """1 · Maximise x * y subject to x + 2 * y = 8. Return (x, y, lam), lam the Lagrange multiplier."""
    raise NotImplementedError


def nearest_on_line():
    """2 · Minimise x**2 + y**2 subject to 3 * x + 4 * y = 10. Return (x, y, lam), with the Lagrangian
    written as x**2 + y**2 - lam * (3 * x + 4 * y - 10)."""
    raise NotImplementedError


def best_value(B):
    """3 · The largest value of 10 * log(1 + x) + 6 * log(1 + y) subject to x + y = B, x >= 0, y >= 0,
    for a budget B from 1 to 40."""
    raise NotImplementedError


def best_lambda(B):
    """3 (continued) · The multiplier of the budget in question 3 at budget B."""
    raise NotImplementedError


def corner_case():
    """4 · Maximise 10 * log(1 + x) + log(1 + y) subject to x + y = 4, x >= 0, y >= 0.
    Return (x, y, lam)."""
    raise NotImplementedError


def project_budget(v, B):
    """5 · The point x >= 0 with sum(x) = B nearest to the vector v (a 1-D NumPy array)."""
    raise NotImplementedError


# 6 · In two or three sentences: read the multiplier of question 1 as a price. If the constraint were
# x + 2 * y = 9 instead of 8, about how much larger would the best x * y be, and why only about?
price_reading = "Write your answer here."

_ = np  # here for your answers
