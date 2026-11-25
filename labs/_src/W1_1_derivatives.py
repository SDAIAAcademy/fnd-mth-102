# %% [markdown]
# # Lab 1.1 · Three ways to a derivative
#
# Week 1 · Session 1 (Derivatives as local lines) · 105 min.
#
# You compute the same three derivatives three ways: by hand with the rules, with symbolic algebra
# (SymPy), and numerically with finite differences. Then you sweep the finite-difference step from
# 0.1 down to 1e-12, watch the error fall and rise again, and explain why. Last, you test how far
# a tangent line and a quadratic stay close to the real curve.
#
# **Reference state.** Nothing to build on: the case file `data/case.json` (synthetic) holds the
# charity's completion model and the district enrolment curves.
#
# **AI level: AI-tutor.** A tutor may give hints, not answers (instruction text in `labs/TUTOR.md`).
#
# **Tasks**
# - [ ] 1 · Derive three derivatives by hand and code them (15')
# - [ ] 2 · Get the same derivatives from SymPy and compare (15')
# - [ ] 3 · Write forward and central differences (15')
# - [ ] 4 · Sweep the step from 0.1 to 1e-12 and plot the error (20')
# - [ ] 5 · Explain the V: truncation against rounding, and the best step (15')
# - [ ] 6 · Test how far the tangent line and the quadratic hold (15')
# - [ ] 7 · Write three findings in the lab log (10')
# - [ ] ✱ Stretch · Compute the complex-step derivative and see why it has no rounding V

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

import numpy as np
import mth as U

C = U.case()
M = C["completion_model"]
D1 = C["districts"]["D1"]
print("completion model:", M)
print("district D1 enrolment curve:", D1)

# %% [markdown]
# ## The three functions
#
# All three come from the charity's (synthetic) planning files.
#
# | Function | Meaning | Point |
# |---|---|---|
# | `p(a) = sigmoid(w_a*a + w_h*4 + b)` | chance a learner with attendance share `a` and 4 weekly hours completes | `a = 0.6` |
# | `E(x) = k * log(1 + x/s)` | enrolments in district D1 for an outreach budget of `x` thousand | `x = 20` |
# | `q(t) = 12 * t * exp(-0.25*t)` | hours of practice a typical learner logs in week `t` | `t = 3` |
#
# `sigmoid(z) = 1 / (1 + exp(-z))`, and `sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))`.

# %%
def p(a):
    return U.sigmoid(M["w_a"] * a + M["w_h"] * 4.0 + M["b"])


def E(x):
    return D1["k"] * np.log1p(x / D1["s"])


def q(t):
    return 12.0 * t * np.exp(-0.25 * t)


FUNCS = {"p": (p, 0.6), "E": (E, 20.0), "q": (q, 3.0)}
for name, (f, x0) in FUNCS.items():
    print(f"{name}({x0}) = {f(x0):.6f}")

# %% [markdown]
# ## 1 · Derivatives by hand (15')
#
# Work on paper first: the chain rule for `p`, the chain rule for `E`, the product rule for `q`.
# Then code each derivative.
#
# ✏️ **Exercise** · Write `dp`, `dE` and `dq`, the exact derivatives.

# %%
def dp(a):
    # BEGIN SOLUTION
    s = p(a)
    return M["w_a"] * s * (1 - s)
    # END SOLUTION


def dE(x):
    # BEGIN SOLUTION
    return D1["k"] / (D1["s"] + x)
    # END SOLUTION


def dq(t):
    # BEGIN SOLUTION
    return 12.0 * np.exp(-0.25 * t) * (1 - 0.25 * t)
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    hand = {"p": dp(0.6), "E": dE(20.0), "q": dq(3.0)}
    for name, (f, x0) in FUNCS.items():
        assert abs(hand[name] - U.central_diff(f, x0, 1e-5)) < 1e-6, name
except NotImplementedError:
    hand = U.reference("w1_hand")
for k, v in hand.items():
    print(f"{k}' = {v:.6f}")

# %% [markdown]
# ## 2 · The same derivatives from SymPy (15')
#
# A computer algebra system applies the same rules you did, exactly. `sp.diff` differentiates,
# `sp.lambdify` turns the result into a NumPy function.
#
# ✏️ **Exercise** · Build the three expressions with SymPy symbols and return their derivatives.

# %%
import sympy as sp

a, x, t = sp.symbols("a x t", real=True)


def symbolic_derivatives():
    # BEGIN SOLUTION
    p_expr = 1 / (1 + sp.exp(-(M["w_a"] * a + M["w_h"] * 4 + M["b"])))
    E_expr = D1["k"] * sp.log(1 + x / D1["s"])
    q_expr = 12 * t * sp.exp(-sp.Rational(1, 4) * t)
    return {"p": sp.diff(p_expr, a), "E": sp.diff(E_expr, x), "q": sp.diff(q_expr, t)}
    # END SOLUTION

# %%
try:
    sym = symbolic_derivatives()
    sym_val = {
        "p": float(sym["p"].subs(a, 0.6)),
        "E": float(sym["E"].subs(x, 20.0)),
        "q": float(sym["q"].subs(t, 3.0)),
    }
    print("SymPy says q'(t) =", sp.simplify(sym["q"]))
except NotImplementedError:
    sym_val = U.reference("w1_hand")
for k in hand:
    print(f"{k}': hand {hand[k]:.10f}   symbolic {sym_val[k]:.10f}   difference {abs(hand[k] - sym_val[k]):.1e}")

# %% [markdown]
# **Look at** the differences: they are at the level of the last digits a float can hold. Hand
# rules and symbolic algebra are the same mathematics; only the bookkeeping differs.

# %% [markdown]
# ## 3 · Forward and central differences (15')
#
# The forward difference `(f(x+h) - f(x)) / h` is the slope of a secant. The central difference
# `(f(x+h) - f(x-h)) / (2h)` uses a point on each side.
#
# ✏️ **Exercise** · Write both.

# %%
def forward_diff(f, x0, h):
    # BEGIN SOLUTION
    return (f(x0 + h) - f(x0)) / h
    # END SOLUTION


def central_diff(f, x0, h):
    # BEGIN SOLUTION
    return (f(x0 + h) - f(x0 - h)) / (2 * h)
    # END SOLUTION

# %%
try:
    fd_ok = abs(forward_diff(E, 20.0, 1e-4) - hand["E"]) < 1e-3 and abs(central_diff(E, 20.0, 1e-4) - hand["E"]) < 1e-7
    assert fd_ok
except NotImplementedError:
    print("(safety net: using the course's own finite differences)")
    forward_diff = lambda f, x0, h: (f(x0 + h) - f(x0)) / h  # noqa: E731
    central_diff = lambda f, x0, h: (f(x0 + h) - f(x0 - h)) / (2 * h)  # noqa: E731
print("E'(20) forward, h=1e-4:", forward_diff(E, 20.0, 1e-4))
print("E'(20) central, h=1e-4:", central_diff(E, 20.0, 1e-4))
print("E'(20) exact:           ", hand["E"])

# %% [markdown]
# ## 4 · Sweep the step (20')
#
# ✏️ **Exercise** · For each `h` in `HS`, compute the absolute error of the forward and of the
# central difference against the exact derivative, for each of the three functions. Return a
# dict `{name: {"forward": [...], "central": [...]}}`.

# %%
HS = np.logspace(-1, -12, 23)


def sweep():
    # BEGIN SOLUTION
    res = {}
    for name, (f, x0) in FUNCS.items():
        res[name] = {
            "forward": [abs(forward_diff(f, x0, h) - hand[name]) for h in HS],
            "central": [abs(central_diff(f, x0, h) - hand[name]) for h in HS],
        }
    return res
    # END SOLUTION

# %%
try:
    errors = sweep()
    assert len(errors["E"]["central"]) == len(HS)
except NotImplementedError:
    errors = U.reference("w1_sweep")
plt = U.plt()
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
for ax, name in zip(axes, FUNCS):
    ax.loglog(HS, np.maximum(errors[name]["forward"], 1e-17), "o-", ms=3, label="forward")
    ax.loglog(HS, np.maximum(errors[name]["central"], 1e-17), "s-", ms=3, label="central")
    ax.set_title(f"{name}'"), ax.set_xlabel("step h")
axes[0].set_ylabel("absolute error"), axes[0].legend()
fig.tight_layout()
fig.savefig(U.out("W1_1_error_sweep.png"), dpi=120)
print("saved out/W1_1_error_sweep.png")
for name in FUNCS:
    for kind in ("forward", "central"):
        e = np.array(errors[name][kind])
        i = int(np.argmin(e))
        print(f"{name:2} {kind:8} best h = {HS[i]:.0e}   error there = {e[i]:.1e}   error at h=1e-12: {e[-1]:.1e}")

# %% [markdown]
# ## 5 · Explain the V (15')
#
# Two errors pull in opposite directions:
#
# - **Truncation.** Taylor's theorem: `f(x+h) = f(x) + h f'(x) + h²/2 f''(x) + ...`. The forward
#   difference keeps an error of about `h/2 |f''|`; in the central one the `h²` terms cancel and
#   the error is about `h²/6 |f'''|`. Smaller `h`, smaller truncation error.
# - **Rounding.** Each value of `f` carries a relative error of about machine epsilon
#   (`eps ≈ 2.2e-16`). Dividing a difference of two nearly equal numbers by `h` magnifies it
#   to about `eps |f| / h`. Smaller `h`, larger rounding error.
#
# Balancing the two gives a best step near `sqrt(eps)` for forward differences and near
# `eps^(1/3)` for central ones.
#
# ✏️ **Exercise** · Return the two theoretical best steps, and the measured best step of each
# kind averaged over the three functions (geometric mean of the three best `h`).

# %%
def best_steps():
    # BEGIN SOLUTION
    eps = np.finfo(float).eps
    measured = {}
    for kind in ("forward", "central"):
        bests = [HS[int(np.argmin(errors[n][kind]))] for n in FUNCS]
        measured[kind] = float(np.exp(np.mean(np.log(bests))))
    return {"theory_forward": float(np.sqrt(eps)), "theory_central": float(np.cbrt(eps)),
            "measured_forward": measured["forward"], "measured_central": measured["central"]}
    # END SOLUTION

# %%
try:
    steps = best_steps()
    assert 1e-10 < steps["measured_forward"] < 1e-6 and 1e-7 < steps["measured_central"] < 1e-3
except NotImplementedError:
    steps = U.reference("w1_steps")
for k, v in steps.items():
    print(f"{k:17} {v:.1e}")

# %% [markdown]
# ## 6 · How far does a local approximation hold? (15')
#
# Around `x0 = 20` the tangent line is `T1(x) = E(x0) + E'(x0)(x - x0)`, and the quadratic adds
# `E''(x0)/2 (x - x0)²` with `E''(x) = -k / (s + x)²`.
#
# ✏️ **Exercise** · Write `tangent(xx)` and `quadratic(xx)`, then return, for each, the largest
# budget change `dx` in `DXS` whose relative error stays below 1%.

# %%
X0 = 20.0
DXS = [1, 2, 5, 10, 15, 20, 30, 40, 60]


def tangent(xx):
    # BEGIN SOLUTION
    return E(X0) + dE(X0) * (xx - X0)
    # END SOLUTION


def quadratic(xx):
    # BEGIN SOLUTION
    d2 = -D1["k"] / (D1["s"] + X0) ** 2
    return tangent(xx) + 0.5 * d2 * (xx - X0) ** 2
    # END SOLUTION


def reach(approx):
    # BEGIN SOLUTION
    ok = [dx for dx in DXS if abs(approx(X0 + dx) - E(X0 + dx)) / E(X0 + dx) < 0.01]
    return max(ok) if ok else 0
    # END SOLUTION

# %%
try:
    holds = {"tangent": reach(tangent), "quadratic": reach(quadratic)}
    rows = [(dx, E(X0 + dx), tangent(X0 + dx), quadratic(X0 + dx)) for dx in DXS]
except NotImplementedError:
    holds = U.reference("w1_reach")
    rows = []
for dx, e, t1, t2 in rows:
    print(f"dx={dx:>3}  E={e:7.3f}  tangent={t1:7.3f} ({abs(t1 - e) / e:6.2%})  quadratic={t2:7.3f} ({abs(t2 - e) / e:6.2%})")
print("within 1%: tangent up to dx =", holds["tangent"], "| quadratic up to dx =", holds["quadratic"])

# %% [markdown]
# **Look at** the signs: the tangent line always overestimates `E`, because the curve bends down
# (`E'' < 0`). The planning team's "each extra thousand brings E'(20) learners" holds for small
# changes only.

# %% [markdown]
# ## 7 · The lab log (10')
#
# ✏️ **Exercise** · Write three findings, each with a number from your run: one on symbolic
# against hand derivatives, one on the best step, one on the reach of the tangent line.

# %%
def findings():
    # BEGIN SOLUTION
    return [
        f"Hand and SymPy derivatives agree to {max(abs(hand[k] - sym_val[k]) for k in hand):.0e}: the same rules, exact.",
        f"Central differences are best near h = {steps['measured_central']:.0e} and forward ones near "
        f"{steps['measured_forward']:.0e}; below that, rounding error grows as h shrinks.",
        f"The tangent line of E at 20 thousand stays within 1% up to a change of {holds['tangent']} thousand, "
        f"the quadratic up to {holds['quadratic']}.",
    ]
    # END SOLUTION

# %%
try:
    log = findings()
except NotImplementedError:
    log = ["(write your three findings here)"]
U.out("W1_1_log.md").write_text("# Lab 1.1 log\n\n" + "\n".join(f"- {s}" for s in log) + "\n", encoding="utf-8")
print("\n".join(log))

# %% [markdown]
# ## ✱ Stretch · The complex step
#
# For a function that accepts complex numbers, `Im f(x + ih) / h` gives `f'(x)` with no
# subtraction at all, so no cancellation. Try `h = 1e-20` on `q` and compare with `dq(3)`.

# %%
h = 1e-20
cs = (12.0 * (3 + 1j * h) * np.exp(-0.25 * (3 + 1j * h))).imag / h
print("complex step:", cs, "| exact:", hand["q"], "| error:", abs(cs - hand["q"]))

# %% [markdown]
# ## Checkpoint questions
#
# 1. A forward difference with `h = 1e-13` gives a worse answer than one with `h = 1e-7`. Why?
#    <details><summary>Answer</summary>Rounding error grows like eps/h: at 1e-13 the
#    subtraction f(x+h) - f(x) has lost most of its significant digits.</details>
# 2. Why does the central difference reach a smaller error than the forward one?
#    <details><summary>Answer</summary>Its h² terms cancel, so its truncation error falls like
#    h² instead of h, and the balance with rounding sits at a lower error.</details>
# 3. The tangent line of `E` at 20 overestimates `E` at 40. What property of `E` says so before
#    you compute anything?
#    <details><summary>Answer</summary>E is concave there (E'' &lt; 0): the curve bends below
#    every tangent line.</details>
#
# **Carry forward.** `central_diff` comes back in every lab as the referee for derivatives you
# compute by hand or in code. The log is in `out/W1_1_log.md`.
