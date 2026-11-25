"""The starter interface of the scalar reverse-mode engine (Lab 4.1; the capstone starts from your
finished version of it).

Copy this file into your capstone repository as engine.py, or paste in the class you completed
in Lab 4.1. Check it with:

    import engine_tests
    from engine import Value
    engine_tests.run_tests(Value)
"""
from __future__ import annotations

import math


class Value:
    """A scalar that remembers how it was computed, so that backward() can apply the chain rule."""

    def __init__(self, data: float, parents: tuple = (), op: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._parents = parents          # the Values this one was computed from
        self._backward = lambda: None    # sends self.grad to the parents, times the local derivatives
        self.op = op                     # for printing and debugging only

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"

    # ---- the two operations everything else is built from
    def __add__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    # ---- the backward pass
    def backward(self) -> None:
        """Topological order of the graph that ends here, then each node's _backward in reverse order."""
        raise NotImplementedError

    def zero_grad(self) -> None:
        raise NotImplementedError

    # ---- more operations, each with its local derivative
    def __pow__(self, k: float):
        raise NotImplementedError

    def exp(self):
        raise NotImplementedError

    def log(self):
        raise NotImplementedError

    def tanh(self):
        raise NotImplementedError

    def relu(self):
        raise NotImplementedError

    # ---- built from the others: no new derivative needed
    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1.0 if isinstance(other, Value) else 1.0 / other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return (-self) + other

    def __rtruediv__(self, other):
        return Value(other) * self ** -1.0

    def sigmoid(self):
        return 1.0 / (1.0 + (-self).exp())


_ = math  # math.exp, math.log and math.tanh are what the local derivatives need
