"""The course's reference engine: the solution of Lab 4.1, used by the lab's safety net.

Trainers and the lab's safety net use it; participants write their own (engine_starter.py).
"""
from __future__ import annotations

import math


class Value:
    def __init__(self, data, parents=(), op=""):
        self.data = float(data)
        self.grad = 0.0
        self._parents = parents
        self._backward = lambda: None
        self.op = op

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def _topo(self):
        order, seen, stack = [], set(), [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if id(p) not in seen:
                    stack.append((p, False))
        return order

    def backward(self):
        order = self._topo()
        self.grad = 1.0
        for node in reversed(order):
            node._backward()

    def zero_grad(self):
        for node in self._topo():
            node.grad = 0.0

    # built from the others, as in engine_starter.py
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


def _pow(self, k):
    out = Value(self.data ** k, (self,), f"**{k}")

    def _backward():
        self.grad += k * self.data ** (k - 1) * out.grad
    out._backward = _backward
    return out


def _exp(self):
    e = math.exp(self.data)
    out = Value(e, (self,), "exp")

    def _backward():
        self.grad += e * out.grad
    out._backward = _backward
    return out


def _log(self):
    out = Value(math.log(self.data), (self,), "log")

    def _backward():
        self.grad += out.grad / self.data
    out._backward = _backward
    return out


def _tanh(self):
    t = math.tanh(self.data)
    out = Value(t, (self,), "tanh")

    def _backward():
        self.grad += (1 - t * t) * out.grad
    out._backward = _backward
    return out


def _relu(self):
    out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

    def _backward():
        self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
    out._backward = _backward
    return out


Value.__pow__, Value.exp, Value.log, Value.tanh, Value.relu = _pow, _exp, _log, _tanh, _relu
