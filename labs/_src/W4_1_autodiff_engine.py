# %% [markdown]
# # Lab 4.1 · A scalar reverse-mode engine
#
# Week 4 · Session 1 (A reverse-mode engine) · 105 min.
#
# You write the engine that did your Lab 3.1 by hand: a `Value` class that records every
# operation, and a `backward()` that walks the graph in reverse topological order applying the
# chain rule. A test suite tells you what works; finite differences and a framework's autograd
# (PyTorch) referee the gradients. The capstone trains a network with this engine.
#
# **Reference state.** The starter interface is `engine_starter.py`; the test suite is
# `engine_tests.py`. Nothing from earlier labs is needed (your Lab 3.1 values are one of the tests).
#
# **AI level: AI-tutor.** Hints, not answers (`labs/TUTOR.md`). The oral check asks you to trace
# a backward pass through this code, so write it yourself.
#
# **Tasks**
# - [ ] 1 · Read the interface and run the test suite (5')
# - [ ] 2 · Implement addition, multiplication and backward (25')
# - [ ] 3 · Add power, exp, log, tanh and ReLU (20')
# - [ ] 4 · Pass the finite-difference checks on random expressions (15')
# - [ ] 5 · Compare your gradients with a framework's (15')
# - [ ] 6 · Fix gradient accumulation for a reused node (15')
# - [ ] 7 · Write three findings in the lab log (10')
# - [ ] ✱ Stretch · Add a dot-product node with its vector-Jacobian product

# %%
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    REPO_URL = "https://github.com/SDAIAAcademy/fnd-mth-102.git"
    if not Path("/content/fnd-mth-102").exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, "/content/fnd-mth-102"], check=True)
    os.chdir("/content/fnd-mth-102/labs")
else:
    while not Path("_tools").exists() and Path.cwd() != Path.cwd().parent:
        os.chdir("..")  # solutions/ notebooks run from labs/
sys.path.insert(0, os.getcwd())

import math
import engine_tests
import mth as U

# %% [markdown]
# ## 1 · The interface and the tests (5')
#
# Open `engine_starter.py` and read it: a `Value` holds `data`, `grad`, its `_parents` and a
# `_backward` closure. The operations at the bottom (`-`, `/`, `sigmoid` and the reversed ones)
# are already written in terms of `+`, `*`, `**` and `exp`: you write those and they follow.
#
# The class below is the same interface. Run the suite on it now: every test fails.

# %% [markdown]
# ## 2 · Addition, multiplication and backward (25')
#
# Each operation creates a new `Value` and attaches a `_backward` that **adds** the output's
# `grad`, times the local derivative, to each input's `grad`. `backward()` sets its own `grad`
# to 1, orders the graph so that every node comes after its inputs (a topological sort, written
# with a loop and an explicit stack, not recursion: the suite has a 5,000-node chain), and calls
# `_backward` on the nodes in reverse order.
#
# ✏️ **Exercise** · Write `__add__`, `__mul__`, `backward` and `zero_grad`. A plain number on
# either side becomes a `Value` first.

# %%
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
        # BEGIN SOLUTION
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
        # END SOLUTION

    def __mul__(self, other):
        # BEGIN SOLUTION
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
        # END SOLUTION

    def _topo(self):
        # BEGIN SOLUTION
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
        # END SOLUTION

    def backward(self):
        # BEGIN SOLUTION
        order = self._topo()
        self.grad = 1.0
        for node in reversed(order):
            node._backward()
        # END SOLUTION

    def zero_grad(self):
        # BEGIN SOLUTION
        for node in self._topo():
            node.grad = 0.0
        # END SOLUTION

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

# %%
a, b = Value(2.0), Value(-3.0)
try:
    c = a * b + a
    c.backward()
    print("c =", c, "| a =", a, "| b =", b)
except NotImplementedError:
    print("(task 2 not done yet)")

# %% [markdown]
# ## 3 · Power, exp, log, tanh and ReLU (20')
#
# ✏️ **Exercise** · Each needs its value and its local derivative: `d(x**k)/dx = k x**(k-1)`,
# `d exp(x)/dx = exp(x)`, `d log(x)/dx = 1/x`, `d tanh(x)/dx = 1 - tanh(x)**2`, and for ReLU 1
# where `x > 0`, else 0 (the engine's choice at exactly 0: 0).

# %%
def _pow(self, k):
    # BEGIN SOLUTION
    out = Value(self.data ** k, (self,), f"**{k}")

    def _backward():
        self.grad += k * self.data ** (k - 1) * out.grad
    out._backward = _backward
    return out
    # END SOLUTION


def _exp(self):
    # BEGIN SOLUTION
    e = math.exp(self.data)
    out = Value(e, (self,), "exp")

    def _backward():
        self.grad += e * out.grad
    out._backward = _backward
    return out
    # END SOLUTION


def _log(self):
    # BEGIN SOLUTION
    out = Value(math.log(self.data), (self,), "log")

    def _backward():
        self.grad += out.grad / self.data
    out._backward = _backward
    return out
    # END SOLUTION


def _tanh(self):
    # BEGIN SOLUTION
    t = math.tanh(self.data)
    out = Value(t, (self,), "tanh")

    def _backward():
        self.grad += (1 - t * t) * out.grad
    out._backward = _backward
    return out
    # END SOLUTION


def _relu(self):
    # BEGIN SOLUTION
    out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

    def _backward():
        self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
    out._backward = _backward
    return out
    # END SOLUTION


Value.__pow__, Value.exp, Value.log, Value.tanh, Value.relu = _pow, _exp, _log, _tanh, _relu

# %%
passed, total = engine_tests.run_tests(Value)
if passed == total:
    Engine = Value
else:
    # Safety net: the later tasks run on the course's reference engine until yours passes
    sys.path.insert(0, str(Path("solutions").resolve()))
    from engine_reference import Value as Engine
    print("(safety net: tasks 4 to 6 use the reference engine in solutions/engine_reference.py)")

# %% [markdown]
# ## 4 · Finite-difference checks on random expressions (15')
#
# `engine_tests.random_expression(Value, seed)` builds a random expression of three inputs from
# your operations. ✏️ **Exercise** · For seeds 100 to 149, compare your engine's gradient with
# central differences and return the largest relative error over all 50 expressions.

# %%
def worst_fd_error(V):
    # BEGIN SOLUTION
    worst = 0.0
    for seed in range(100, 150):
        build, xs = engine_tests.random_expression(V, seed)
        vs = [V(x) for x in xs]
        out = build(vs)
        out.backward()
        g = [v.grad for v in vs]
        f = lambda arr: build(list(arr))  # noqa: E731
        g_num = U.central_diff(f, xs)
        worst = max(worst, U.rel_err(g, g_num))
    return worst
    # END SOLUTION

# %%
try:
    fd_worst = worst_fd_error(Engine)
except NotImplementedError:
    fd_worst = U.reference("w4_fd_worst")
print(f"largest relative error against central differences over 50 random expressions: {fd_worst:.1e}")
assert fd_worst < 1e-6

# %% [markdown]
# ## 5 · Compare with a framework (15')
#
# A framework's automatic differentiation (here PyTorch autograd, in float64) is a second
# referee, without the step-size error of finite differences.
#
# ✏️ **Exercise** · Rebuild the Lab 3.1 loss with PyTorch tensors that have
# `requires_grad=True`, call `.backward()`, and return PyTorch's gradient for `(w1, w2, b)`.
# Then the cell compares it with your engine's, and runs `engine_tests.compare_with_framework`
# on ten random expressions.

# %%
import torch

T3 = U.case()["two_learners"]


def torch_two_learner_grad():
    # BEGIN SOLUTION
    X = torch.tensor(T3["x"], dtype=torch.float64)
    Y = torch.tensor(T3["y"], dtype=torch.float64)
    w = torch.tensor(T3["w"], dtype=torch.float64, requires_grad=True)
    b = torch.tensor(T3["b"], dtype=torch.float64, requires_grad=True)
    p = torch.sigmoid(X @ w + b)
    L = -(Y * torch.log(p) + (1 - Y) * torch.log(1 - p)).mean()
    L.backward()
    return [w.grad[0].item(), w.grad[1].item(), b.grad.item()]
    # END SOLUTION


def engine_two_learner_grad(V):
    w1, w2, b = V(T3["w"][0]), V(T3["w"][1]), V(T3["b"])
    total = 0
    for (x1, x2), y in zip(T3["x"], T3["y"]):
        p = (w1 * x1 + w2 * x2 + b).sigmoid()
        total = total + (-(y * p.log() + (1 - y) * (1 - p).log()))
    (total / 2).backward()
    return [w1.grad, w2.grad, b.grad]

# %%
g_engine = engine_two_learner_grad(Engine)
tol = 1e-10
try:
    g_torch = torch_two_learner_grad()
except NotImplementedError:
    g_torch, tol = U.reference("w4_torch_grad"), 1e-6  # the reference is stored to eight digits
print("engine :", [f"{g:.8f}" for g in g_engine])
print("PyTorch:", [f"{g:.8f}" for g in g_torch])
fw_two = U.rel_err(g_engine, g_torch)
fw_random = engine_tests.compare_with_framework(Engine)
print(f"relative difference, Lab 3.1 loss: {fw_two:.1e}; ten random expressions, worst: {fw_random:.1e}")
assert fw_two < tol and fw_random < 1e-10

# %% [markdown]
# **Look at** the two referees: against finite differences the engine agrees to about 1e-9 or
# 1e-10 (the step-size error); against autograd to 1e-13 or better (both compute exact
# derivatives, so only the order of floating-point operations differs).

# %% [markdown]
# ## 6 · Gradient accumulation (15')
#
# Here is a multiplication written by someone who used `=` where `+=` belongs.

# %%
def mul_overwrite(self, other):
    other = other if isinstance(other, Engine) else Engine(other)
    out = Engine(self.data * other.data, (self, other), "*")

    def _backward():
        self.grad = other.data * out.grad   # overwrites what other paths sent
        other.grad = self.data * out.grad
    out._backward = _backward
    return out

# %% [markdown]
# ✏️ **Exercise** · Install `mul_overwrite` on a copy of the engine, run the suite, and return
# the names of the tests that fail. Then explain, in the log, why each of them fails.

# %%
def failing_with_overwrite():
    # BEGIN SOLUTION
    original = Engine.__mul__
    Engine.__mul__ = mul_overwrite
    failed = []
    try:
        for t in engine_tests.TESTS:
            try:
                ok = t(Engine)
            except Exception:
                ok = False
            if not ok:
                failed.append(t.__name__[2:])
    finally:
        Engine.__mul__ = original
    return failed
    # END SOLUTION

# %%
try:
    broken_tests = failing_with_overwrite()
except NotImplementedError:
    broken_tests = U.reference("w4_overwrite_fails")
print("tests that fail with an overwriting multiplication:", broken_tests)
assert "same_node_twice" in broken_tests

# %% [markdown]
# ## 7 · The lab log (10')
#
# ✏️ **Exercise** · Three findings with numbers from your run: the suite, the two referees, the
# accumulation bug.

# %%
def findings():
    # BEGIN SOLUTION
    return [
        f"The engine passes {passed} of {total} tests, including the 5,000-node chain (an iterative topological sort).",
        f"Against central differences the worst relative error over 50 random expressions is {fd_worst:.0e}; "
        f"against PyTorch autograd {max(fw_two, fw_random):.0e}: finite differences carry step error, autograd does not.",
        f"Overwriting instead of adding in multiplication breaks {len(broken_tests)} tests ({', '.join(broken_tests)}): "
        "every graph in which a node reaches the output along more than one path.",
    ]
    # END SOLUTION

# %%
try:
    log = findings()
except NotImplementedError:
    log = ["(write your three findings here)"]
U.out("W4_1_log.md").write_text("# Lab 4.1 log\n\n" + "\n".join(f"- {s}" for s in log) + "\n", encoding="utf-8")
print("\n".join(log))
print("\nFor the capstone: copy your Value class and the five functions of task 3 into engine.py.")

# %% [markdown]
# ## ✱ Stretch · A dot-product node
#
# A node `dot(ws, xs)` for lists of Values returns `sum(w * x)` as **one** node whose backward
# sends `x * out.grad` to each `w` and `w * out.grad` to each `x`: the vector-Jacobian product of
# the dot product. It makes a layer's graph far smaller than one node per multiplication. Write it
# and test it against `sum(w * x for ...)`.

# %%
def dot(ws, xs):
    out = Engine(sum(w.data * x.data for w, x in zip(ws, xs)), tuple(ws) + tuple(xs), "dot")

    def _backward():
        for w, x in zip(ws, xs):
            w.grad += x.data * out.grad
            x.grad += w.data * out.grad
    out._backward = _backward
    return out


ws, xs = [Engine(v) for v in (0.5, -1.0, 2.0)], [Engine(v) for v in (1.5, 0.2, -0.3)]
(dot(ws, xs) * 2).backward()
print("dot-node gradients of w:", [round(w.grad, 6) for w in ws])

# %% [markdown]
# ## Checkpoint questions
#
# 1. Why must `_backward` add to `grad` instead of setting it?
#    <details><summary>Answer</summary>A node that feeds several others receives one
#    contribution per path; the chain rule sums them.</details>
# 2. Why does `backward()` need a topological order?
#    <details><summary>Answer</summary>A node may pass its adjoint on only after every node that
#    uses it has sent its contribution; reverse topological order guarantees that.</details>
# 3. Finite differences agree with your engine to 1e-9, PyTorch to 1e-13. Which one would reveal
#    a gradient that is wrong by 1e-7?
#    <details><summary>Answer</summary>Both, but only PyTorch with a clear margin: 1e-7 sits
#    close to finite differences' own error.</details>
#
# **Carry forward.** The capstone trains a two-layer network with this engine: keep the class.
# The log is in `out/W4_1_log.md`.
