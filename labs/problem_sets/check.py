"""Auto-check a problem set by the properties its answers must have.

    cd labs
    uv run python problem_sets/check.py PS1            # checks problem_sets/PS1.py
    uv run python problem_sets/check.py PS1 my_ps1.py  # checks another file

The checker holds no answer key: it tests each function you wrote against finite differences,
shapes, symmetry or the definition of the quantity asked for. A written answer is only checked
for being there; the trainer marks its content. Problem sets are AI-off for the first attempt.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def cdiff(f, x, h=1e-6):
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        return (f(x + h) - f(x - h)) / (2 * h)
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e.flat[i] = h
        g.flat[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def close(a, b, tol=1e-5) -> bool:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(np.all(np.abs(a - b) <= tol * (1 + np.abs(b))))


def written(text) -> bool:
    return isinstance(text, str) and len(text.split()) >= 12 and "write" not in text.lower()[:20]


CHECKS: dict[str, list] = {}


def check(ps: str, label: str):
    def deco(fn):
        CHECKS.setdefault(ps, []).append((label, fn))
        return fn
    return deco


# ------------------------------------------------------------------ PS1 · week 1
@check("PS1", "1 · derivative of x^3 e^(-x)")
def _(m):
    f = lambda x: x ** 3 * math.exp(-x)  # noqa: E731
    return all(close(m.d1(x), cdiff(f, x)) for x in (-1.0, 0.5, 2.0, 3.0, 6.0))


@check("PS1", "2 · derivative of log(1 + x^2) / (1 + x)")
def _(m):
    f = lambda x: math.log(1 + x * x) / (1 + x)  # noqa: E731
    return all(close(m.d2(x), cdiff(f, x)) for x in (0.0, 0.7, 1.5, 4.0))


@check("PS1", "3 · tangent line of sqrt(1 + x) at x = 3")
def _(m):
    f = lambda x: math.sqrt(1 + x)  # noqa: E731
    slope, intercept = m.tangent_at_3()
    return close(slope, cdiff(f, 3.0)) and close(intercept, f(3.0) - cdiff(f, 3.0) * 3.0)


@check("PS1", "4 · value, slope and curvature of exp(-x^2) at x = 0.5")
def _(m):
    f = lambda x: math.exp(-x * x)  # noqa: E731
    v, d, dd = m.taylor_terms()
    second = (f(0.5 + 1e-4) - 2 * f(0.5) + f(0.5 - 1e-4)) / 1e-8
    return close(v, f(0.5)) and close(d, cdiff(f, 0.5)) and close(dd, second, 1e-4)


@check("PS1", "5 · the best central-difference step for exp(x) at x = 2")
def _(m):
    hs = [10.0 ** -k for k in range(1, 13)]
    errs = [abs((math.exp(2 + h) - math.exp(2 - h)) / (2 * h) - math.exp(2)) for h in hs]
    return m.best_step() == hs[int(np.argmin(errs))]


@check("PS1", "6 · written: why a smaller step is not always better")
def _(m):
    return written(m.why_not_smaller)


# ------------------------------------------------------------------ PS2 · week 2
_f2 = lambda v: v[0] ** 2 * v[1] + 3 * v[1] ** 3  # noqa: E731


@check("PS2", "1 · gradient of x^2 y + 3y^3")
def _(m):
    return all(close(m.grad_f(x, y), cdiff(_f2, [x, y])) for x, y in ((1.0, 2.0), (-0.5, 0.3), (2.0, -1.0)))


@check("PS2", "2 · gradient of log(1 + exp(w . z))")
def _(m):
    z = np.array([1.0, -2.0, 0.5])
    g = lambda w: math.log1p(math.exp(float(w @ z)))  # noqa: E731
    return all(close(m.grad_g(np.array(w)), cdiff(g, w)) for w in ([0.1, 0.2, -0.3], [1.0, -1.0, 2.0]))


@check("PS2", "3 · directional derivative at (1, 2) along (3/5, 4/5)")
def _(m):
    u = np.array([0.6, 0.8])
    return close(m.directional(), float(cdiff(_f2, [1.0, 2.0]) @ u))


@check("PS2", "4 · Jacobian of (xy, x + y^2, sin x)")
def _(m):
    F = lambda v: np.array([v[0] * v[1], v[0] + v[1] ** 2, math.sin(v[0])])  # noqa: E731
    def jfd(v):
        v = np.asarray(v, float)
        return np.column_stack([(F(v + e) - F(v - e)) / 2e-6 for e in np.eye(2) * 1e-6])
    return all(close(m.jac_F(x, y), jfd([x, y])) for x, y in ((1.0, 2.0), (0.3, -0.7)))


@check("PS2", "5 · unit direction of steepest descent at (1, 2)")
def _(m):
    d = np.asarray(m.steepest_descent_direction(), float)
    g = cdiff(_f2, [1.0, 2.0])
    return close(np.linalg.norm(d), 1.0) and close(d, -g / np.linalg.norm(g))


@check("PS2", "6 · written: why a relative error")
def _(m):
    return written(m.why_relative)


# ------------------------------------------------------------------ PS3 · week 3
@check("PS3", "1 · gradient of (xy + y)^2, y reused")
def _(m):
    q = lambda v: (v[0] * v[1] + v[1]) ** 2  # noqa: E731
    return all(close(m.grad_q(x, y), cdiff(q, [x, y])) for x, y in ((2.0, 3.0), (-0.5, 1.5), (0.3, -2.0)))


@check("PS3", "2 · gradient of (a + b) c + exp(a)")
def _(m):
    g = lambda v: (v[0] + v[1]) * v[2] + math.exp(v[0])  # noqa: E731
    return all(close(m.grad_g(*v), cdiff(g, v)) for v in ([0.1, 0.2, 0.3], [1.0, -2.0, 0.5]))


@check("PS3", "3 · gradient of a one-learner logistic loss")
def _(m):
    x = np.array([0.3, -1.2, 2.0])
    def loss(th):
        p = 1 / (1 + math.exp(-(float(th[:3] @ x) + th[3])))
        return -math.log(1 - p)
    ok = True
    for th in ([0.5, 0.1, -0.2, 0.05], [-1.0, 0.4, 0.3, -0.2]):
        gw, gb = m.logistic_grad(np.array(th[:3]), th[3])
        num = cdiff(loss, th)
        ok &= close(gw, num[:3]) and close(gb, num[3])
    return ok


@check("PS3", "4 · the adjoint of z")
def _(m):
    ok = True
    for z in (-2.0, 0.3, 1.7):
        for y in (0, 1):
            f = lambda zz: -(y * math.log(1 / (1 + math.exp(-zz))) + (1 - y) * math.log(1 - 1 / (1 + math.exp(-zz))))  # noqa: E731
            ok &= close(m.z_adjoint(1 / (1 + math.exp(-z)), y), cdiff(f, z))
    return ok


@check("PS3", "5 · passes per mode")
def _(m):
    return all(m.passes(n, k) == {"forward": n, "backward": k} for n, k in ((1000, 3), (2, 50), (7, 7)))


@check("PS3", "6 · written: why the adjoint is a sum")
def _(m):
    return written(m.why_sum)


# ------------------------------------------------------------------ PS4 · week 4
_RNG = np.random.default_rng(4)
_A = _RNG.normal(size=(2, 3))
_X = _RNG.normal(size=3)
_G = _RNG.normal(size=2)


@check("PS4", "1 · vector-Jacobian product of A x, with respect to x")
def _(m):
    return close(m.vjp_matvec_x(_A, _G), cdiff(lambda x: float(_G @ (_A @ x)), _X))


@check("PS4", "2 · gradient of A x with respect to A")
def _(m):
    return close(m.vjp_matvec_A(_X, _G), cdiff(lambda A: float(_G @ (A @ _X)), _A))


@check("PS4", "3 · vector-Jacobian product of tanh")
def _(m):
    g = np.array([0.3, -1.0, 2.0])
    return close(m.vjp_tanh(_X, g), cdiff(lambda x: float(g @ np.tanh(x)), _X))


@check("PS4", "4 · Jacobian of a composition")
def _(m):
    H = lambda x: np.array([math.sin(x[0] * x[1]), (x[0] * x[1]) * (x[0] + x[1]), math.exp(x[0] + x[1])])  # noqa: E731
    x = np.array([0.4, -0.7])
    J = np.column_stack([(H(x + e) - H(x - e)) / 2e-6 for e in np.eye(2) * 1e-6])
    return close(m.jac_composite(x), J)


@check("PS4", "5 · a sin node for the engine")
def _(m):
    sys.path.insert(0, str(HERE.parent / "solutions"))
    from engine_reference import Value
    sin = m.make_sin(Value)
    a = Value(0.8)
    out = sin(a * a) * a
    out.backward()
    f = lambda v: math.sin(v * v) * v  # noqa: E731
    return close(out.data, f(0.8)) and close(a.grad, cdiff(f, 0.8))


@check("PS4", "6 · written: why vector-Jacobian products")
def _(m):
    return written(m.why_vjp)


# ------------------------------------------------------------------ PS5 · week 5
_H5 = np.array([[3.0, 1.0], [1.0, 3.0]])


@check("PS5", "1 · plain gradient descent")
def _(m):
    w = np.array([1.0, -2.0])
    for _ in range(7):
        w = w - 0.1 * (_H5 @ w)
    return close(m.gd(lambda v: _H5 @ v, np.array([1.0, -2.0]), 0.1, 7), w)


@check("PS5", "2 · the largest stable rate")
def _(m):
    r = float(m.largest_stable_rate())
    def runs(lr):
        w = np.array([1.0, 1.0])
        for _ in range(400):
            w = w - lr * (_H5 @ w)
        return np.linalg.norm(w) < 1.0
    return runs(r * 0.98) and not runs(r * 1.02)


@check("PS5", "3 · one momentum step")
def _(m):
    w, v, g = np.array([1.0, 2.0]), np.array([0.5, -0.5]), np.array([0.2, 0.4])
    w2, v2 = m.momentum_step(w, v, g, 0.1, 0.9)
    return close(v2, 0.9 * v + g) and close(w2, w - 0.1 * (0.9 * v + g))


@check("PS5", "4 · a mini-batch gradient")
def _(m):
    rng = np.random.default_rng(55)
    A, b, w = rng.normal(size=(40, 3)), rng.normal(size=40), rng.normal(size=3)
    idx = np.array([3, 7, 11, 20, 33])
    f = lambda v: 0.5 * float(np.mean((A[idx] @ v - b[idx]) ** 2))  # noqa: E731
    return close(m.minibatch_grad(A, b, w, idx), cdiff(f, w))


@check("PS5", "5 · diagnosing four runs")
def _(m):
    ref = 10.0
    good = list(100 * 0.6 ** np.arange(40) + ref)
    slow = list(100 * 0.99 ** np.arange(40) + ref)
    noisy = list(100 * 0.6 ** np.arange(40) + ref + np.r_[np.zeros(20), np.tile([0.0, 0.9], 10)])
    bad = list(100 * 1.3 ** np.arange(40))
    return (m.diagnose(good, ref), m.diagnose(slow, ref), m.diagnose(noisy, ref), m.diagnose(bad, ref)) == (
        "converged", "slow", "noisy", "diverged")


@check("PS5", "6 · written: noise and a decaying rate")
def _(m):
    return written(m.why_noise)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ps = sys.argv[1] if len(sys.argv) > 1 else "PS1"
    path = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / f"{ps}.py"
    spec = importlib.util.spec_from_file_location(ps, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    score = 0
    for label, fn in CHECKS[ps]:
        try:
            ok = bool(fn(m))
            why = "" if ok else "does not match the check"
        except NotImplementedError:
            ok, why = False, "not done yet"
        except Exception as e:  # a wrong answer may also crash: report it, keep going
            ok, why = False, f"{type(e).__name__}: {e}"
        score += ok
        print(f"{'PASS' if ok else 'FAIL'}  {label}{'  (' + why + ')' if why else ''}")
    print(f"{ps}: {score} of {len(CHECKS[ps])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
