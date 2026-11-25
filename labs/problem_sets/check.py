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
