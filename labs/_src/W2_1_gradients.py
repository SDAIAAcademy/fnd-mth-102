# %% [markdown]
# # Lab 2.1 · Gradients and a Jacobian, checked
#
# Week 2 · Session 1 (Gradients and Jacobians) · 105 min.
#
# The charity's completion model has two inputs, so it has two partial derivatives and a
# gradient. Its district enrolment map has five inputs and five outputs, so it has a Jacobian.
# You derive both, write a gradient checker you will reuse all term, predict the effect of moving
# budget between districts, and catch a wrong partial derivative in a colleague's notes.
#
# **Reference state.** Nothing to build on: `data/case.json` (synthetic). If you kept your
# `central_diff` from Lab 1.1, compare it with the one you write here.
#
# **AI level: AI-tutor.** Hints, not answers (`labs/TUTOR.md`).
#
# **Tasks**
# - [ ] 1 · Derive the partial derivatives of the completion model (15')
# - [ ] 2 · Build the gradient and test the steepest direction (15')
# - [ ] 3 · Write a gradient checker for any function of n inputs (15')
# - [ ] 4 · Build two Jacobians and check them (20')
# - [ ] 5 · Predict a budget shift with the Jacobian and compare (15')
# - [ ] 6 · Find the planted wrong partial derivative (15')
# - [ ] 7 · Write three findings in the lab log (10')
# - [ ] ✱ Stretch · Check the symmetry of second partial derivatives numerically

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
DIST = C["districts"]
K = np.array([DIST[d]["k"] for d in DIST])
S = np.array([DIST[d]["s"] for d in DIST])
print("completion model:", M)
print("enrolment curves k:", K, "s:", S)

# %% [markdown]
# ## 1 · Partial derivatives of the completion model (15')
#
# `p(a, h) = sigmoid(w_a*a + w_h*h + b)`: attendance share `a` and weekly hours `h`. A partial
# derivative holds every other input still. With `s = p(a, h)`, the chain rule gives each one.
#
# ✏️ **Exercise** · Write `dp_da(a, h)` and `dp_dh(a, h)`.

# %%
def p(a, h):
    return U.sigmoid(M["w_a"] * a + M["w_h"] * h + M["b"])


def dp_da(a, h):
    # BEGIN SOLUTION
    s = p(a, h)
    return M["w_a"] * s * (1 - s)
    # END SOLUTION


def dp_dh(a, h):
    # BEGIN SOLUTION
    s = p(a, h)
    return M["w_h"] * s * (1 - s)
    # END SOLUTION

# %%
A0, H0 = 0.7, 5.0
try:
    partials = [dp_da(A0, H0), dp_dh(A0, H0)]
    assert abs(partials[0] - U.central_diff(lambda a: p(a, H0), A0)) < 1e-7
    assert abs(partials[1] - U.central_diff(lambda h: p(A0, h), H0)) < 1e-7
except NotImplementedError:
    partials = U.reference("w2_partials")
print(f"p({A0}, {H0}) = {p(A0, H0):.4f}   dp/da = {partials[0]:.4f}   dp/dh = {partials[1]:.4f}")

# %% [markdown]
# **Look at** the two numbers: one more unit of attendance share (from 0 to 1, the whole range)
# is worth far more than one more weekly hour. Their ratio is `w_a / w_h` at every point. Why?

# %% [markdown]
# ## 2 · The gradient and the steepest direction (15')
#
# The gradient stacks the partials: `grad p = (dp/da, dp/dh)`. For a unit direction `u`, the
# directional derivative is `grad p · u`. It is largest when `u` points along the gradient.
#
# ✏️ **Exercise** · Compute the directional derivative numerically (central difference along
# `u`) for 360 directions, one per degree, and return the angle in degrees where it is largest,
# and the largest value.

# %%
def steepest_numerically():
    # BEGIN SOLUTION
    best_angle, best = 0, -np.inf
    for deg in range(360):
        u = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
        dd = U.central_diff(lambda t: p(A0 + t * u[0], H0 + t * u[1]), 0.0)
        if dd > best:
            best_angle, best = deg, dd
    return best_angle, float(best)
    # END SOLUTION

# %%
grad = np.array(partials)
try:
    angle, top = steepest_numerically()
except NotImplementedError:
    angle, top = U.reference("w2_steepest")
print(f"largest directional derivative {top:.4f} at {angle} degrees")
print(f"gradient direction {np.degrees(np.arctan2(grad[1], grad[0])):.1f} degrees, gradient norm {np.linalg.norm(grad):.4f}")

# %% [markdown]
# **Look at** the two lines: the best direction is the gradient's (to the nearest degree), and the
# largest rate equals the length of the gradient.

# %% [markdown]
# ## 3 · A gradient checker (15')
#
# ✏️ **Exercise** · `check_gradient(f, grad_f, x)` takes a function of a vector, its claimed
# gradient and a point. It returns the relative error
# `|g_claimed - g_numeric| / max(|g_claimed|, |g_numeric|)` with central differences of step
# `1e-6` per coordinate. You will use this checker in weeks 3, 4 and 7 and in the capstone.

# %%
def check_gradient(f, grad_f, x, h=1e-6):
    # BEGIN SOLUTION
    x = np.asarray(x, dtype=float)
    g_num = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = h
        g_num[i] = (f(x + e) - f(x - e)) / (2 * h)
    g = np.asarray(grad_f(x), dtype=float)
    return float(np.linalg.norm(g - g_num) / max(np.linalg.norm(g), np.linalg.norm(g_num), 1e-12))
    # END SOLUTION

# %%
f_vec = lambda v: p(v[0], v[1])  # noqa: E731
g_vec = lambda v: np.array([dp_da(v[0], v[1]), dp_dh(v[0], v[1])])  # noqa: E731
try:
    err = check_gradient(f_vec, g_vec, [A0, H0])
    assert err < 1e-7, err
except NotImplementedError:
    check_gradient = lambda f, g, x, h=1e-6: U.rel_err(g(np.asarray(x, float)), U.central_diff(f, x, h))  # noqa: E731
    err = U.reference("w2_check")
print(f"relative error of the hand gradient: {err:.1e}")

# %% [markdown]
# ## 4 · Two Jacobians (20')
#
# A map with `n` inputs and `m` outputs has an `m × n` Jacobian: row `i` is the gradient of
# output `i`.
#
# - **District enrolments.** `E(x) = k * log(1 + x / s)`, one district per coordinate (five in,
#   five out).
# - **Learner summary.** `F(a, h) = (p(a, h), 12 * a * h)`: the completion chance and the hours
#   actually attended in a 12-week term (two in, two out).
#
# ✏️ **Exercise** · Write `jac_E(x)` (5 × 5), `jac_F(a, h)` (2 × 2) and `jacobian_fd(F, x)`, a
# numerical Jacobian by central differences, column by column.

# %%
def E(x):
    return K * np.log1p(np.asarray(x, dtype=float) / S)


def F(v):
    a, h = v
    return np.array([p(a, h), 12 * a * h])


def jac_E(x):
    # BEGIN SOLUTION
    return np.diag(K / (S + np.asarray(x, dtype=float)))
    # END SOLUTION


def jac_F(a, h):
    # BEGIN SOLUTION
    return np.array([[dp_da(a, h), dp_dh(a, h)], [12 * h, 12 * a]])
    # END SOLUTION


def jacobian_fd(func, x, h=1e-6):
    # BEGIN SOLUTION
    x = np.asarray(x, dtype=float)
    cols = []
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = h
        cols.append((func(x + e) - func(x - e)) / (2 * h))
    return np.column_stack(cols)
    # END SOLUTION

# %%
X0 = np.array([30.0, 25.0, 25.0, 20.0, 20.0])  # this year's split, thousands; total 120
try:
    JE, JF = jac_E(X0), jac_F(A0, H0)
    eE = U.rel_err(JE, jacobian_fd(E, X0))
    eF = U.rel_err(JF, jacobian_fd(F, [A0, H0]))
    assert eE < 1e-7 and eF < 1e-7, (eE, eF)
except NotImplementedError:
    ref = U.reference("w2_jacobians")
    JE, JF, eE, eF = np.array(ref["JE"]), np.array(ref["JF"]), 0.0, 0.0
np.set_printoptions(precision=4, suppress=True)
print("Jacobian of E at this year's split:\n", JE, f"\nrelative error {eE:.1e}")
print("Jacobian of F at (0.7, 5):\n", JF, f"\nrelative error {eF:.1e}")

# %% [markdown]
# **Look at** the zeros of the first Jacobian: one district's budget does not change another's
# enrolments in this model. A real outreach campaign can spill over; the model says it cannot.

# %% [markdown]
# ## 5 · Predict a budget shift (15')
#
# The programme lead proposes moving 5 thousand from D1 to D4. The Jacobian predicts the change
# of every district's enrolments: `E(x0 + d) ≈ E(x0) + J d`.
#
# ✏️ **Exercise** · Return the predicted and the actual change of total enrolments for the
# shift, and the same for a shift of 20 thousand.

# %%
def shift_effects():
    # BEGIN SOLUTION
    res = {}
    for amount in (5.0, 20.0):
        d = np.array([-amount, 0, 0, amount, 0])
        res[amount] = {"predicted": float((JE @ d).sum()), "actual": float(E(X0 + d).sum() - E(X0).sum())}
    return res
    # END SOLUTION

# %%
try:
    shifts = shift_effects()
except NotImplementedError:
    shifts = {float(k): v for k, v in U.reference("w2_shifts").items()}
for amount, v in shifts.items():
    print(f"move {amount:4.0f} from D1 to D4: predicted {v['predicted']:+.2f}, actual {v['actual']:+.2f} enrolments")

# %% [markdown]
# **Look at** the signs: on enrolments alone the move loses learners, because D1's curve is
# steeper at 30 than D4's at 20. Week 8 asks whether enrolments are the right thing to count.

# %% [markdown]
# ## 6 · A wrong partial derivative (15')
#
# Last term's notes hold this gradient of the completion model with respect to its three
# parameters `(w_a, w_h, b)` at one learner. It was used to tune the model. Check it.

# %%
def notes_grad(theta, a, h):
    """From last term's notes: gradient of p with respect to (w_a, w_h, b)."""
    w_a, w_h, b = theta
    s = U.sigmoid(w_a * a + w_h * h + b)
    return np.array([s * (1 - s) * a, s * (1 - s) * a * h, s * (1 - s)])

# %% [markdown]
# ✏️ **Exercise** · Use `check_gradient` on `notes_grad` at `theta = (4.0, 0.3, -3.5)`,
# `a = 0.7`, `h = 5`. Return the index of the wrong component (0, 1 or 2) by comparing component
# by component, and a corrected gradient function.

# %%
THETA = np.array([M["w_a"], M["w_h"], M["b"]])
p_theta = lambda th: U.sigmoid(th[0] * A0 + th[1] * H0 + th[2])  # noqa: E731


def find_wrong_component():
    # BEGIN SOLUTION
    g = notes_grad(THETA, A0, H0)
    num = U.central_diff(p_theta, THETA)
    return int(np.argmax(np.abs(g - num) / np.maximum(np.abs(num), 1e-12)))
    # END SOLUTION


def fixed_grad(theta, a, h):
    # BEGIN SOLUTION
    w_a, w_h, b = theta
    s = U.sigmoid(w_a * a + w_h * h + b)
    return np.array([s * (1 - s) * a, s * (1 - s) * h, s * (1 - s)])
    # END SOLUTION

# %%
print(f"notes gradient, relative error: {check_gradient(p_theta, lambda th: notes_grad(th, A0, H0), THETA):.1e}")
try:
    wrong = find_wrong_component()
    fixed_err = check_gradient(p_theta, lambda th: fixed_grad(th, A0, H0), THETA)
    assert wrong == 1 and fixed_err < 1e-7
except NotImplementedError:
    wrong, fixed_err = U.reference("w2_wrong")
print(f"wrong component: {wrong} ({['w_a', 'w_h', 'b'][wrong]}); corrected gradient, relative error: {fixed_err:.1e}")

# %% [markdown]
# ## 7 · The lab log (10')
#
# ✏️ **Exercise** · Three findings, each with a number from your run: the steepest direction,
# the budget shift, the wrong partial.

# %%
def findings():
    # BEGIN SOLUTION
    return [
        f"The completion chance rises fastest at {angle} degrees from the attendance axis, at {top:.3f} per unit: "
        "the gradient's direction and length.",
        f"Moving 5 thousand from D1 to D4 changes total enrolments by {shifts[5.0]['actual']:+.2f}; the Jacobian "
        f"predicted {shifts[5.0]['predicted']:+.2f}. For 20 thousand the prediction is off by "
        f"{abs(shifts[20.0]['predicted'] - shifts[20.0]['actual']):.2f}.",
        f"The notes' partial for {['w_a', 'w_h', 'b'][wrong]} carried an extra factor a; the checker flagged it at once.",
    ]
    # END SOLUTION

# %%
try:
    log = findings()
except NotImplementedError:
    log = ["(write your three findings here)"]
U.out("W2_1_log.md").write_text("# Lab 2.1 log\n\n" + "\n".join(f"- {s}" for s in log) + "\n", encoding="utf-8")
print("\n".join(log))

# %% [markdown]
# ## ✱ Stretch · Symmetric second partials
#
# For a smooth function, `d/dh (dp/da) = d/da (dp/dh)`. Check it with central differences of your
# partials.

# %%
try:
    mixed1 = U.central_diff(lambda h: dp_da(A0, h), H0, 1e-5)
    mixed2 = U.central_diff(lambda a: dp_dh(a, H0), A0, 1e-5)
    print(f"d/dh dp/da = {mixed1:.6f}   d/da dp/dh = {mixed2:.6f}")
except NotImplementedError:
    print("(finish task 1 first)")

# %% [markdown]
# ## Checkpoint questions
#
# 1. The gradient of the completion model at a learner is (0.47, 0.035). In which direction of the
#    (attendance, hours) plane does the chance rise fastest?
#    <details><summary>Answer</summary>Along (0.47, 0.035) itself, almost straight along the
#    attendance axis; the rate is the gradient's length, about 0.47.</details>
# 2. Why is the Jacobian of the district map diagonal?
#    <details><summary>Answer</summary>Each district's enrolments depend only on its own budget, so
#    every cross partial is zero.</details>
# 3. A gradient check returns a relative error of 0.08 on one component and 1e-9 on the others.
#    What do you conclude?
#    <details><summary>Answer</summary>That component's formula is wrong; a correct gradient
#    gives errors near 1e-7 or below in every component.</details>
#
# **Carry forward.** `check_gradient` is reused in Labs 3.1, 4.1 and 7.1. The log is in
# `out/W2_1_log.md`.
