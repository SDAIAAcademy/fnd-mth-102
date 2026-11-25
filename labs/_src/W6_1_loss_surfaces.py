# %% [markdown]
# # Lab 6.1 · Six loss surfaces
#
# Week 6 · Session 1 (Convexity and curvature) · 105 min.
#
# Six two-parameter loss surfaces, each with a gradient-descent trace from a given start and
# step. You classify their critical points from the Hessian, test each surface for convexity,
# trace descent below, at and above the largest stable step, start on and off a saddle's ridge,
# and annotate every plot. Your annotations are then compared with the reference: this is the
# form of the evidence for outcome LO2.
#
# **Reference state.** Nothing to build on: the surfaces are defined below.
#
# **AI level: AI-tutor.** Hints, not answers (`labs/TUTOR.md`).
#
# **Tasks**
# - [ ] 1 · Plot the six surfaces as contour maps (10')
# - [ ] 2 · Classify critical points from the Hessian (20')
# - [ ] 3 · Test each surface for convexity (15')
# - [ ] 4 · Trace gradient descent below, at and above the stable step (20')
# - [ ] 5 · Start on and off the saddle's ridge (10')
# - [ ] 6 · Annotate each plot against the reference (20')
# - [ ] 7 · Write three findings in the lab log (10')
# - [ ] ✱ Stretch · Take Newton steps on the valley and compare with gradient descent

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

# The six surfaces: (function of w = (x, y), trace start, step, number of steps, plotting box)
SURFACES = {
    "bowl": (lambda w: 0.5 * (w[0] ** 2 + 2 * w[1] ** 2), (2.0, 2.0), 0.4, 60, (-3, 3, -3, 3)),
    "narrow": (lambda w: 0.5 * (w[0] ** 2 + 25 * w[1] ** 2), (3.0, 1.0), 0.085, 60, (-4, 4, -1.5, 1.5)),
    "valley": (lambda w: (1 - w[0]) ** 2 + 5 * (w[1] - w[0] ** 2) ** 2, (-1.0, 2.0), 0.02, 1500, (-1.5, 1.8, -0.5, 2.5)),
    "saddle": (lambda w: w[0] ** 2 - w[1] ** 2 + 0.25 * w[1] ** 4, (1.0, 0.0), 0.1, 200, (-2, 2, -2.2, 2.2)),
    "bumpy": (lambda w: 0.1 * w[0] ** 2 + 0.5 * np.sin(2 * w[0]) + w[1] ** 2, (3.0, 1.0), 0.1, 300, (-4, 4, -2, 2)),
    "plateau": (lambda w: 1 - np.exp(-(w[0] ** 2 + w[1] ** 2) / 8), (8.0, 8.0), 0.5, 300, (-10, 10, -10, 10)),
}
for name, (f, start, lr, n, box) in SURFACES.items():
    print(f"{name:8} start {start}  step {lr}  steps {n}  f(start) = {f(np.array(start)):.4g}")

# %% [markdown]
# ## 1 · Contour maps (10')
#
# The cell draws the six surfaces. Before running it, write down for each one what you expect to
# see: one minimum, several, a ridge, a flat region.

# %%
plt = U.plt()


def draw(traces=None, notes=None, name="W6_1_surfaces.png"):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (key, (f, start, lr, n, box)) in zip(axes.flat, SURFACES.items()):
        xs, ys = np.linspace(box[0], box[1], 160), np.linspace(box[2], box[3], 160)
        Xg, Yg = np.meshgrid(xs, ys)
        Z = f(np.array([Xg, Yg]))
        ax.contour(Xg, Yg, Z, levels=20, linewidths=0.7)
        if traces and key in traces:
            p = np.array(traces[key])
            p = p[np.all(np.isfinite(p), axis=1) & (np.abs(p).max(axis=1) < 1e3)]
            ax.plot(p[:, 0], p[:, 1], ".-", color="tab:red", ms=2, lw=1)
            ax.set_xlim(box[0], box[1]), ax.set_ylim(box[2], box[3])
        ax.set_title(key if not notes else f"{key}: {notes[key]}", fontsize=9)
    fig.tight_layout()
    fig.savefig(U.out(name), dpi=110)
    plt.close(fig)


draw()
print("saved out/W6_1_surfaces.png")

# %% [markdown]
# ## 2 · Critical points and the Hessian (20')
#
# At a critical point the gradient is zero. The Hessian (the matrix of second partial
# derivatives) tells what kind: all eigenvalues positive, a minimum; all negative, a maximum;
# both signs, a saddle; a zero eigenvalue, the test cannot tell ("flat").
#
# ✏️ **Exercise** · Write `hessian_fd(f, w)` with central differences of step 1e-4 (four
# evaluations per entry) and `classify(H)`, which returns `"minimum"`, `"maximum"`, `"saddle"` or
# `"flat"` (any eigenvalue within 1e-6 of zero).

# %%
def hessian_fd(f, w, h=1e-4):
    # BEGIN SOLUTION
    w = np.asarray(w, dtype=float)
    n = w.size
    H = np.zeros((n, n))
    E = np.eye(n) * h
    for i in range(n):
        for j in range(n):
            H[i, j] = (f(w + E[i] + E[j]) - f(w + E[i] - E[j]) - f(w - E[i] + E[j]) + f(w - E[i] - E[j])) / (4 * h * h)
    return (H + H.T) / 2
    # END SOLUTION


def classify(H):
    # BEGIN SOLUTION
    e = np.linalg.eigvalsh(H)
    if np.any(np.abs(e) < 1e-6):
        return "flat"
    if np.all(e > 0):
        return "minimum"
    if np.all(e < 0):
        return "maximum"
    return "saddle"
    # END SOLUTION

# %%
POINTS = {"bowl": (0, 0), "narrow": (0, 0), "valley": (1, 1), "saddle": (0, 0), "saddle, side": (0, 2 ** 0.5)}
found = {}
for label, pt in POINTS.items():
    f = SURFACES[label.split(",")[0]][0]
    try:
        H = hessian_fd(f, pt)
        found[label] = (np.round(np.linalg.eigvalsh(H), 3).tolist(), classify(H))
    except NotImplementedError:
        found[label] = U.reference("w6_points")[label]
    print(f"{label:13} at {tuple(round(v, 3) for v in pt)}: Hessian eigenvalues {found[label][0]} -> {found[label][1]}")

# %% [markdown]
# ## 3 · Convexity (15')
#
# A twice-differentiable function is convex on a region when its Hessian has no negative
# eigenvalue anywhere in it.
#
# ✏️ **Exercise** · `looks_convex(f, box)` evaluates the Hessian on a 15 × 15 grid of the
# surface's plotting box and returns `True` when the smallest eigenvalue found is above -1e-6.
# A grid can only refute convexity, never prove it: say so in your log.

# %%
def looks_convex(f, box):
    # BEGIN SOLUTION
    smallest = np.inf
    for x in np.linspace(box[0], box[1], 15):
        for y in np.linspace(box[2], box[3], 15):
            smallest = min(smallest, np.linalg.eigvalsh(hessian_fd(f, (x, y)))[0])
    return bool(smallest > -1e-6)
    # END SOLUTION

# %%
try:
    convex = {k: looks_convex(v[0], v[4]) for k, v in SURFACES.items()}
except NotImplementedError:
    convex = U.reference("w6_convex")
print(convex)

# %% [markdown]
# ## 4 · Below, at and above the stable step (20')
#
# On `narrow`, `0.5 (x² + 25 y²)`, the largest curvature is 25, so the edge is `2 / 25 = 0.08`.
#
# ✏️ **Exercise** · `gd_trace(f, start, lr, steps)` returns the list of points visited (using
# `U.central_diff` for the gradient), stopping early if a point is no longer finite. Run it on
# `narrow` from (3, 1) for 60 steps at 0.05, 0.079 and 0.081, and report the final loss of each.

# %%
def gd_trace(f, start, lr, steps):
    # BEGIN SOLUTION
    w = np.array(start, dtype=float)
    path = [w.copy()]
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(steps):
            w = w - lr * U.central_diff(f, w)
            path.append(w.copy())
            if not np.all(np.isfinite(w)):
                break
    return path
    # END SOLUTION

# %%
f_narrow = SURFACES["narrow"][0]
try:
    edge = {lr: float(f_narrow(gd_trace(f_narrow, (3.0, 1.0), lr, 60)[-1])) for lr in (0.05, 0.079, 0.081)}
except NotImplementedError:
    edge = {float(k): v for k, v in U.reference("w6_edge").items()}
for lr, v in edge.items():
    print(f"narrow, step {lr}: loss after 60 steps {v:.4g}")

# %% [markdown]
# ## 5 · On and off the saddle's ridge (10')
#
# `saddle` is `x² - y² + y⁴/4`: a saddle at the origin and two minima at `y = ±√2`.
#
# ✏️ **Exercise** · Trace 200 steps of 0.1 from (1, 0), exactly on the ridge `y = 0`, and from
# (1, 0.001). Return the two end points.

# %%
def ridge_runs():
    # BEGIN SOLUTION
    f = SURFACES["saddle"][0]
    return gd_trace(f, (1.0, 0.0), 0.1, 200)[-1].tolist(), gd_trace(f, (1.0, 0.001), 0.1, 200)[-1].tolist()
    # END SOLUTION

# %%
try:
    on_ridge, off_ridge = ridge_runs()
except NotImplementedError:
    on_ridge, off_ridge = U.reference("w6_ridge")
print("from (1, 0):    ends at", np.round(on_ridge, 4))
print("from (1, 0.001): ends at", np.round(off_ridge, 4))

# %% [markdown]
# **Look at** the first end point: gradient descent stopped at a saddle because the gradient's
# y-component is exactly zero on the ridge. A thousandth of noise was enough to escape. Stochastic
# gradients supply that noise in practice.

# %% [markdown]
# ## 6 · Annotate the six plots (20')
#
# For each surface, run its own trace (start, step and number of steps from `SURFACES`) and
# annotate it with three things:
#
# - `convex`: your answer from task 3;
# - `outcome`: `"diverged"` (not finite or beyond 1,000), `"plateau"` (moved less than 0.01 in
#   total), `"saddle"` (ended where the Hessian says saddle) or `"minimum"`;
# - `global`: for a minimum, whether its loss is within 1e-3 of the lowest value of the surface
#   on a 400 × 400 grid of its box; `None` otherwise.
#
# ✏️ **Exercise** · Write `annotate()` returning `{surface: {"convex", "outcome", "global"}}`
# and the traces. The cell compares your annotations with the reference and draws them.

# %%
def annotate():
    # BEGIN SOLUTION
    notes, traces = {}, {}
    for key, (f, start, lr, n, box) in SURFACES.items():
        path = gd_trace(f, start, lr, n)
        traces[key] = [list(map(float, p)) for p in path]
        end = path[-1]
        if not np.all(np.isfinite(end)) or np.abs(end).max() > 1e3:
            outcome, glob = "diverged", None
        elif np.linalg.norm(end - np.array(start)) < 0.01:
            outcome, glob = "plateau", None
        elif classify(hessian_fd(f, end)) == "saddle":
            outcome, glob = "saddle", None
        else:
            xs, ys = np.meshgrid(np.linspace(box[0], box[1], 400), np.linspace(box[2], box[3], 400))
            outcome, glob = "minimum", bool(f(end) - f(np.array([xs, ys])).min() < 1e-3)
        notes[key] = {"convex": convex[key], "outcome": outcome, "global": glob}
    return notes, traces
    # END SOLUTION

# %%
REFERENCE = U.reference("w6_annotations", quiet=True)
try:
    mine, traces = annotate()
except NotImplementedError:
    mine, traces = REFERENCE, U.reference("w6_traces")
score = 0
for key in SURFACES:
    ok = mine[key] == REFERENCE[key]
    score += ok
    print(f"{key:8} {mine[key]}  {'matches' if ok else 'differs from'} the reference")
print(f"{score} of 6 annotations match the reference")
draw(traces,
     {k: f"{'convex' if v['convex'] else 'non-convex'}, {v['outcome']}" for k, v in mine.items()},
     "W6_1_annotated.png")
print("saved out/W6_1_annotated.png")

# %% [markdown]
# ## 7 · The lab log (10')
#
# ✏️ **Exercise** · Three findings with numbers: the edge on `narrow`, the saddle, and one
# non-convex surface where descent ended somewhere other than the best point.

# %%
def findings():
    # BEGIN SOLUTION
    return [
        f"On narrow the loss after 60 steps was {edge[0.05]:.1e} at step 0.05, {edge[0.079]:.2g} at 0.079 and "
        f"{edge[0.081]:.3g} at 0.081: the edge 2/25 = 0.08 separates shrinking from growing.",
        f"From (1, 0) descent stopped at the saddle {np.round(on_ridge, 3).tolist()}; from (1, 0.001) it reached "
        f"{np.round(off_ridge, 3).tolist()}, a minimum. The Hessian at the origin has eigenvalues {found['saddle'][0]}.",
        "On bumpy, descent from x = 3 settled in a local minimum that is not the lowest point of the surface; "
        "on plateau it did not move, because the gradient there is about 3e-7.",
    ]
    # END SOLUTION

# %%
try:
    log = findings()
except NotImplementedError:
    log = ["(write your three findings here)"]
U.out("W6_1_log.md").write_text("# Lab 6.1 log\n\n" + "\n".join(f"- {s}" for s in log) + "\n", encoding="utf-8")
print("\n".join(log))

# %% [markdown]
# ## ✱ Stretch · Newton steps on the valley
#
# A Newton step solves `H d = -g` and moves by `d`: it uses the curvature that gradient descent
# ignores. Compare how many steps each needs from (-1, 2) to get within 1e-6 of (1, 1).

# %%
f_valley = SURFACES["valley"][0]
w = np.array([-1.0, 2.0])
try:
    for k in range(1, 30):
        w = w + np.linalg.solve(hessian_fd(f_valley, w), -U.central_diff(f_valley, w))
        if np.linalg.norm(w - 1) < 1e-6:
            break
    print(f"Newton reached (1, 1) to within 1e-6 in {k} steps; gradient descent at step 0.02 was given 1,500.")
except NotImplementedError:
    print("(finish task 2 first: Newton steps need your Hessian)")

# %% [markdown]
# ## Checkpoint questions
#
# 1. The Hessian at a critical point has eigenvalues 3 and -0.5. What is the point?
#    <details><summary>Answer</summary>A saddle: the surface curves up along one direction and
#    down along the other.</details>
# 2. Why can a grid of Hessian checks refute convexity but not prove it?
#    <details><summary>Answer</summary>A negative eigenvalue found anywhere refutes it; between
#    the grid points a negative eigenvalue may still hide.</details>
# 3. A convex loss has largest curvature 50. What happens to gradient descent with step 0.05?
#    <details><summary>Answer</summary>It diverges: the edge is 2 / 50 = 0.04.</details>
#
# **Carry forward.** The annotated figure `out/W6_1_annotated.png` is the form of the LO2
# evidence. Week 7 adds a penalty that makes a non-convex-looking loss convex.
