"""The test suite of the scalar reverse-mode engine (Lab 4.1 and the capstone).

    import engine_tests
    engine_tests.run_tests(Value)                # prints PASS or FAIL per test
    engine_tests.compare_with_framework(Value)   # needs PyTorch

The interface your class must offer (see engine_starter.py):

    v = Value(3.0)            v.data, v.grad (0.0 until backward)
    a + b, a * b, -a, a - b, a / b, a ** 2 (a number exponent), with plain numbers on either side
    a.exp(), a.log(), a.tanh(), a.relu(), a.sigmoid()
    loss.backward()           sets .grad of every node that loss depends on
    loss.zero_grad()          sets .grad of every node that loss depends on back to 0
"""
from __future__ import annotations

import math
import random


def _fd(f, xs, i, h=1e-6):
    up, down = list(xs), list(xs)
    up[i] += h
    down[i] -= h
    return (f(up) - f(down)) / (2 * h)


def _close(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def random_expression(Value, seed: int):
    """A random expression of three inputs built from the engine's operations, seeded.

    Returns (build, xs): build(list_of_Values or floats) gives the output, xs the input values.
    """
    rng = random.Random(seed)
    ops = rng.choices(["add", "mul", "sub", "div", "pow", "exp", "log", "tanh", "sigmoid"], k=7)
    xs = [rng.uniform(0.5, 1.5) for _ in range(3)]

    def build(v):
        a, b, c = v
        nodes = [a, b, c]
        for k, op in enumerate(ops):
            x, y = nodes[-1], nodes[k % 3]
            if op == "add":
                r = x + y
            elif op == "mul":
                r = x * y
            elif op == "sub":
                r = x - 0.5 * y
            elif op == "div":
                r = x / (y * y + 1.0)
            elif op == "pow":
                r = (x * x + 0.5) ** 1.5
            elif op == "exp":
                r = (x * 0.3).exp() if hasattr(x, "exp") else math.exp(x * 0.3)
            elif op == "log":
                r = (x * x + 1.0).log() if hasattr(x, "exp") else math.log(x * x + 1.0)
            elif op == "tanh":
                r = x.tanh() if hasattr(x, "exp") else math.tanh(x)
            else:
                r = x.sigmoid() if hasattr(x, "exp") else 1 / (1 + math.exp(-x))
            nodes.append(r)
        return nodes[-1] + nodes[3] * 0.1
    return build, xs


def _grads(Value, build, xs):
    vs = [Value(x) for x in xs]
    out = build(vs)
    out.backward()
    return out.data, [v.grad for v in vs]


def t_add_mul(Value):
    a, b = Value(2.0), Value(-3.0)
    c = a * b + a
    c.backward()
    return _close(c.data, -4.0) and _close(a.grad, -2.0) and _close(b.grad, 2.0)


def t_constants_both_sides(Value):
    a = Value(4.0)
    c = 2 * a - 1 + a / 2 - 3 / a
    c.backward()
    return _close(c.data, 8 + 2 - 1 - 0.75) and _close(a.grad, 2 + 0.5 + 3 / 16)


def t_reused_node(Value):
    x, y = Value(2.0), Value(3.0)
    f = (x + y) * y
    f.backward()
    return _close(f.data, 15.0) and _close(x.grad, 3.0) and _close(y.grad, 8.0)


def t_same_node_twice(Value):
    a = Value(3.0)
    b = a * a
    b.backward()
    return _close(b.data, 9.0) and _close(a.grad, 6.0)


def t_unary_ops(Value):
    ok = True
    for name, f in [("exp", math.exp), ("log", math.log), ("tanh", math.tanh),
                    ("sigmoid", lambda t: 1 / (1 + math.exp(-t))), ("pow", lambda t: t ** 2.5)]:
        for x0 in (0.4, 1.3):
            v = Value(x0)
            out = v ** 2.5 if name == "pow" else getattr(v, name)()
            out.backward()
            ok &= _close(out.data, f(x0)) and _close(v.grad, _fd(lambda xs: f(xs[0]), [x0], 0), 1e-5)
    return ok


def t_relu(Value):
    a, b = Value(1.5), Value(-0.5)
    c = a.relu() + b.relu()
    c.backward()
    return _close(c.data, 1.5) and _close(a.grad, 1.0) and _close(b.grad, 0.0)


def t_random_expressions(Value):
    for seed in range(20):
        build, xs = random_expression(Value, seed)
        out, g = _grads(Value, build, xs)
        f = lambda v: build(v)  # noqa: E731
        if not _close(out, f(xs)):
            return False
        for i in range(3):
            if not _close(g[i], _fd(f, xs, i), 1e-5):
                return False
    return True


def t_two_learner_loss(Value):
    """The loss of Lab 3.1: the engine must reproduce the hand gradient."""
    X, Y = [[0.9, 0.6], [0.4, 0.2]], [1, 0]
    w1, w2, b = Value(1.5), Value(-2.0), Value(0.5)
    total = 0
    for (x1, x2), y in zip(X, Y):
        p = (w1 * x1 + w2 * x2 + b).sigmoid()
        total = total + (-(y * p.log() + (1 - y) * (1 - p).log()))
    L = total / 2
    L.backward()
    return (_close(L.data, 0.76162069, 1e-6) and _close(w1.grad, -0.02070774, 1e-5)
            and _close(w2.grad, -0.03607808, 1e-5) and _close(b.grad, 0.16259912, 1e-5))


def t_deep_graph(Value):
    """A chain of 5,000 operations: the backward pass must not recurse once per node."""
    x = Value(1.0)
    y = x
    for _ in range(5000):
        y = y * 1.0001 + 0.0
    y.backward()
    return _close(x.grad, 1.0001 ** 5000, 1e-6)


def t_zero_grad(Value):
    a = Value(2.0)
    b = a * a
    b.backward()
    b.zero_grad()
    first = a.grad
    b.backward()
    return _close(first, 0.0) and _close(a.grad, 4.0)


TESTS = [t_add_mul, t_constants_both_sides, t_reused_node, t_same_node_twice, t_unary_ops, t_relu,
         t_random_expressions, t_two_learner_loss, t_deep_graph, t_zero_grad]


def run_tests(Value, verbose: bool = True) -> tuple[int, int]:
    passed = 0
    for t in TESTS:
        try:
            ok = bool(t(Value))
            why = ""
        except NotImplementedError:
            ok, why = False, "not implemented yet"
        except RecursionError:
            ok, why = False, "RecursionError: make the topological sort iterative"
        except Exception as e:  # a broken engine may crash anywhere: report and go on
            ok, why = False, f"{type(e).__name__}: {e}"
        passed += ok
        if verbose:
            print(f"{'PASS' if ok else 'FAIL'}  {t.__name__[2:]}{'  (' + why + ')' if why else ''}")
    if verbose:
        print(f"{passed} of {len(TESTS)} tests pass")
    return passed, len(TESTS)


def compare_with_framework(Value, seeds=range(10)) -> float:
    """Largest relative difference between the engine's gradients and PyTorch autograd's (float64)."""
    import torch

    worst = 0.0
    for seed in seeds:
        build, xs = random_expression(Value, seed)
        _, g = _grads(Value, build, xs)
        tx = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in xs]
        build([_Torch(t) for t in tx]).t.backward()
        for gi, ti in zip(g, tx):
            tg = 0.0 if ti.grad is None else ti.grad.item()  # an input the expression does not use
            worst = max(worst, abs(gi - tg) / max(1e-12, abs(gi), abs(tg)))
    return worst


class _Torch:
    """A PyTorch scalar behind the engine's interface, so the same expression builds in both."""

    def __init__(self, t):
        self.t = t

    @staticmethod
    def _u(o):
        return o.t if isinstance(o, _Torch) else o

    def __add__(self, o): return _Torch(self.t + self._u(o))
    def __radd__(self, o): return _Torch(self._u(o) + self.t)
    def __mul__(self, o): return _Torch(self.t * self._u(o))
    def __rmul__(self, o): return _Torch(self._u(o) * self.t)
    def __sub__(self, o): return _Torch(self.t - self._u(o))
    def __rsub__(self, o): return _Torch(self._u(o) - self.t)
    def __truediv__(self, o): return _Torch(self.t / self._u(o))
    def __rtruediv__(self, o): return _Torch(self._u(o) / self.t)
    def __pow__(self, k): return _Torch(self.t ** k)
    def __neg__(self): return _Torch(-self.t)
    def exp(self): return _Torch(self.t.exp())
    def log(self): return _Torch(self.t.log())
    def tanh(self): return _Torch(self.t.tanh())
    def sigmoid(self): return _Torch(self.t.sigmoid())
    def relu(self): return _Torch(self.t.relu())
