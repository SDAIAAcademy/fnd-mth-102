# %% [markdown]
# # Lab 8.1 · A budget allocation two ways
#
# Week 8 · Session 1 (Constraints and prices) · 105 min.
#
# The charity has an outreach budget of 120 thousand to share across its five districts. You
# solve the allocation from the Lagrange conditions and again with projected gradient ascent,
# read the multiplier as the price of the budget, find who the convenient objective leaves out,
# add a floor for every district, and write the recommendation for the finance officer.
#
# **Reference state.** Nothing to build on: the district figures are in `data/case.json`
# (synthetic).
#
# **AI level: AI-tutor.** Hints, not answers (`labs/TUTOR.md`).
#
# **Tasks**
# - [ ] 1 · Write the Lagrangian for two districts and solve it by hand (15')
# - [ ] 2 · Solve the five-district problem from the Lagrange conditions (20')
# - [ ] 3 · Solve it again with projected gradient ascent (20')
# - [ ] 4 · Compare the answers and read the multiplier (15')
# - [ ] 5 · Find who the proxy leaves out and add a floor (20')
# - [ ] 6 · Write the recommendation for the finance officer (15')
# - [ ] ✱ Stretch · Weight each district by need and see how the multiplier moves

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
NAMES = list(C["districts"])
k = np.array([C["districts"][d]["k"] for d in NAMES])
s = np.array([C["districts"][d]["s"] for d in NAMES])
behind = np.array([C["districts"][d]["behind_share"] for d in NAMES])
BUDGET, FLOOR = C["budget"], C["floor"]


def enrolments(x, k=k, s=s):
    """Expected enrolments per district for outreach budgets x (thousands): k log(1 + x / s)."""
    return k * np.log1p(np.asarray(x, dtype=float) / s)


for d, ki, si, bi in zip(NAMES, k, s, behind):
    print(f"{d}: k = {ki:4.0f}, s = {si:4.0f}, share of learners furthest behind {bi:.2f}")
print(f"budget {BUDGET:g} thousand; proposed floor {FLOOR:g} thousand per district")

# %% [markdown]
# ## 1 · Two districts by hand (15')
#
# Take D1 and D2 alone with 40 thousand: maximise `E₁(x₁) + E₂(x₂)` subject to `x₁ + x₂ = 40`.
# The Lagrangian is `E₁(x₁) + E₂(x₂) − λ (x₁ + x₂ − 40)`; setting its derivatives to zero gives
# `k₁ / (s₁ + x₁) = λ = k₂ / (s₂ + x₂)`: at the optimum, the last thousand buys the same number of
# enrolments in both districts.
#
# ✏️ **Exercise** · Solve the two equations on paper, then return `(x₁, x₂, λ)` from
# `two_districts()`. The cell checks the answer against the budget and the equal-margin condition.

# %%
def two_districts():
    # BEGIN SOLUTION
    # 60 (12 + x2) = 52 (10 + x1) with x2 = 40 - x1  =>  112 x1 = 2600
    x1 = 2600 / 112
    x2 = 40 - x1
    return x1, x2, 60 / (10 + x1)
    # END SOLUTION

# %%
try:
    x1, x2, lam2 = two_districts()
except NotImplementedError:
    x1, x2, lam2 = U.reference("w8_two")
print(f"x1 = {x1:.3f}, x2 = {x2:.3f}, lambda = {lam2:.4f}")
print(f"budget used {x1 + x2:.3f}; margins {k[0] / (s[0] + x1):.4f} and {k[1] / (s[1] + x2):.4f}")

# %% [markdown]
# ## 2 · Five districts from the Lagrange conditions (20')
#
# With five districts and budgets that cannot be negative, the conditions say: every funded
# district has margin `k / (s + x) = λ`, so `x = k / λ − s`; a district whose margin at zero is
# below λ gets nothing. So `x(λ) = max(0, k / λ − s)`, and λ is the number that makes the budgets
# add up to the total.
#
# ✏️ **Exercise** · `allocate(budget, floor=0.0, k=k)` finds λ by bisection (the sum of `x(λ)`
# falls as λ rises) with `x(λ) = max(floor, k / λ − s)`, and returns `(x, λ)`.

# %%
def allocate(budget, floor=0.0, k=k):
    # BEGIN SOLUTION
    lo, hi = 1e-9, 1e3
    for _ in range(200):
        lam = (lo * hi) ** 0.5
        x = np.maximum(floor, k / lam - s)
        if x.sum() > budget:
            lo = lam
        else:
            hi = lam
    return np.maximum(floor, k / lam - s), lam
    # END SOLUTION

# %%
try:
    x_lag, lam = allocate(BUDGET)
    x_lag = x_lag.tolist()
except NotImplementedError:
    x_lag, lam = U.reference("w8_lagrange")
x_lag = np.array(x_lag)
for d, xi, ei in zip(NAMES, x_lag, enrolments(x_lag)):
    print(f"{d}: {xi:7.3f} thousand -> {ei:7.3f} enrolments")
print(f"total {x_lag.sum():.3f} thousand, {enrolments(x_lag).sum():.2f} enrolments; lambda = {lam:.4f}")

# %% [markdown]
# ## 3 · Projected gradient ascent (20')
#
# The same problem without the conditions: take a gradient step uphill, then move the result to
# the nearest point that respects the budget (`x ≥ 0`, sum = 120). That nearest point is
# `max(v − θ, 0)` for the one θ that makes it sum to the budget.
#
# ✏️ **Exercise** · Write `project(v, budget)` (find θ by sorting `v`, or by bisection) and
# `ascend(eta, steps=500)`, which starts from an equal split and returns the final `x` and the
# first step at which every entry is within 1e-3 of `x_lag` (or `None`). Run it at the step sizes
# in `ETAS`.

# %%
ETAS = [1, 5, 20, 50]


def project(v, budget):
    # BEGIN SOLUTION
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - budget
    rho = np.nonzero(u - css / np.arange(1, len(v) + 1) > 0)[0][-1]
    return np.maximum(v - css[rho] / (rho + 1), 0.0)
    # END SOLUTION


def ascend(eta, steps=500):
    # BEGIN SOLUTION
    x = np.full(5, BUDGET / 5)
    reached = None
    for t in range(1, steps + 1):
        x = project(x + eta * k / (s + x), BUDGET)
        if reached is None and np.abs(x - x_lag).max() < 1e-3:
            reached = t
    return x, reached
    # END SOLUTION

# %%
try:
    runs = {}
    for eta in ETAS:
        xe, reached = ascend(eta)
        runs[str(eta)] = {"x": xe.tolist(), "reached": reached}
except NotImplementedError:
    runs = U.reference("w8_projected")
for eta, r in runs.items():
    gap = np.abs(np.array(r["x"]) - x_lag).max()
    print(f"step {eta:>3}: within 1e-3 at step {r['reached']}; largest gap after 500 steps {gap:.2e}")

# %% [markdown]
# ## 4 · Compare, and read the multiplier (15')
#
# λ is the price of the budget: the enrolments one more thousand would buy at the optimum. Test
# that reading.
#
# ✏️ **Exercise** · `price_check()` re-solves with a budget of 121 and returns the change in
# total enrolments next to λ.

# %%
def price_check():
    # BEGIN SOLUTION
    x121, _ = allocate(BUDGET + 1)
    return float(enrolments(x121).sum() - enrolments(x_lag).sum()), float(lam)
    # END SOLUTION

# %%
try:
    gain, lam_read = price_check()
except NotImplementedError:
    gain, lam_read = U.reference("w8_price")
print(f"one more thousand adds {gain:.4f} enrolments; lambda = {lam_read:.4f}")

# %% [markdown]
# ## 5 · Who the proxy leaves out (20')
#
# Total enrolments is a convenient objective: one number, easy to maximise. Look at where the
# money went and at the share of each district's learners who are furthest behind.
#
# ✏️ **Exercise** · `with_floor()` re-solves with a floor of `FLOOR` thousand for every district
# and returns a dictionary: the allocation, λ, total enrolments, enrolments of learners furthest
# behind (`behind × enrolments`, summed) for both allocations, and the floor's price in each
# district held at the floor: `λ − k / (s + FLOOR)`, the enrolments the floor costs per thousand
# there. Districts above the floor get `None`.

# %%
def with_floor():
    # BEGIN SOLUTION
    xf, lamf = allocate(BUDGET, FLOOR)
    at_floor = np.isclose(xf, FLOOR)
    return {
        "x": xf.tolist(), "lambda": lamf,
        "total": [float(enrolments(x_lag).sum()), float(enrolments(xf).sum())],
        "behind": [float((behind * enrolments(x_lag)).sum()), float((behind * enrolments(xf)).sum())],
        "floor_price": [float(lamf - k[i] / (s[i] + FLOOR)) if at_floor[i] else None for i in range(5)],
    }
    # END SOLUTION

# %%
try:
    fl = with_floor()
except NotImplementedError:
    fl = U.reference("w8_floor")
print(f"{'':4} {'proxy':>8} {'floor':>8}  behind share")
for i, d in enumerate(NAMES):
    print(f"{d:4} {x_lag[i]:8.3f} {fl['x'][i]:8.3f}  {behind[i]:.2f}")
print(f"total enrolments: {fl['total'][0]:.2f} -> {fl['total'][1]:.2f}")
print(f"enrolments of learners furthest behind: {fl['behind'][0]:.2f} -> {fl['behind'][1]:.2f}")
print(f"lambda with the floor: {fl['lambda']:.4f}")
for d, p in zip(NAMES, fl["floor_price"]):
    if p is not None:
        print(f"the floor in {d} costs {p:.3f} enrolments per thousand")

# %% [markdown]
# **Look at** where the proxy sent the money: the two districts with the largest shares of
# learners furthest behind got almost nothing, because their enrolment curves are flattest. The
# objective did exactly what it was told. Whether that is acceptable is not a mathematical
# question: it is a decision about weights, and it belongs to the programme lead.
#
# ## 6 · The recommendation (15')
#
# ✏️ **Exercise** · Write `recommendation()`: five or six sentences for the finance officer, with
# numbers. State both allocations, what the floor costs and gains, what λ means in money, and who
# decides what. The analyst proposes; the finance officer approves the budget; the programme lead
# decides how districts are weighted. Keep those three roles separate in the text.

# %%
def recommendation():
    # BEGIN SOLUTION
    d4, d5 = fl["floor_price"][3], fl["floor_price"][4]
    return (
        f"Maximising total enrolments alone puts {x_lag[0]:.1f}, {x_lag[1]:.1f} and {x_lag[2]:.1f} thousand into "
        f"D1 to D3, {x_lag[3]:.1f} into D4 and nothing into D5, although D4 and D5 hold the largest shares of "
        "learners furthest behind (0.46 and 0.55). "
        f"A floor of {FLOOR:g} thousand per district lowers expected enrolments from {fl['total'][0]:.1f} to "
        f"{fl['total'][1]:.1f} and raises enrolments of learners furthest behind from {fl['behind'][0]:.1f} to "
        f"{fl['behind'][1]:.1f}. "
        f"At the margin the floor costs {d4:.2f} enrolments per thousand in D4 and {d5:.2f} in D5. "
        f"The budget's multiplier is {fl['lambda']:.2f} with the floor: another thousand of budget would add about "
        f"{fl['lambda']:.1f} enrolments, which is the figure to weigh against any request for more funds. "
        "Segregation of duties: I, the data analyst, propose these two options and their figures; the finance "
        "officer approves the budget and checks the arithmetic; the programme lead decides how much weight the "
        "districts furthest behind should carry, which is a policy choice, not a calculation."
    )
    # END SOLUTION

# %%
try:
    text = recommendation()
except NotImplementedError:
    text = "(write your recommendation here)"
U.out("W8_1_recommendation.md").write_text("# Lab 8.1 recommendation\n\n" + text + "\n", encoding="utf-8")
print(text)

# %% [markdown]
# ## ✱ Stretch · Weight by need
#
# Replace `k` by `k × behind / mean(behind)`: each district's enrolments now count in proportion
# to its share of learners furthest behind. Re-solve and compare the allocation and λ. The weights
# are a policy input: say who should set them.

# %%
try:
    x_need, lam_need = allocate(BUDGET, k=k * behind / behind.mean())
    print("need-weighted allocation:", np.round(x_need, 2).tolist(), f"lambda {lam_need:.4f} (weighted units)")
except NotImplementedError:
    print("(finish task 2 first)")

# %% [markdown]
# ## Checkpoint questions
#
# 1. At the optimum of a budget allocation, λ = 2.5 enrolments per thousand. Roughly what does
#    a cut of 2 thousand cost?
#    <details><summary>Answer</summary>About 5 enrolments: the multiplier is the rate, valid for
#    small changes.</details>
# 2. Why does a district with a very flat enrolment curve receive nothing?
#    <details><summary>Answer</summary>Its margin at zero, k / s, is already below λ: its first
#    thousand buys less than the last thousand anywhere else.</details>
# 3. Projected gradient ascent with step 50 never settled. What would you change?
#    <details><summary>Answer</summary>Lower the step: as in week 5, too large a step overshoots
#    the optimum on every iteration.</details>
#
# **Carry forward.** The capstone brief is next: train a two-layer network with your own engine
# and report every run, including the ones that failed.
