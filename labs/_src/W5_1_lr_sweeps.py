# %% [markdown]
# # Lab 5.1 · Learning-rate sweeps
#
# Week 5 · Session 1 (Descent and loss curves) · 105 min.
#
# Before the charity trusts an optimiser on its own data, its analyst practises on public data
# whose answer is known. You fit a linear model to the **Airfoil Self-Noise** data (UCI Machine
# Learning Repository, CC BY 4.0: 1,503 wind-tunnel measurements, five features, sound level in
# decibels) with three variants of gradient descent, sweep the learning rate of each, and write an
# experiment log that diagnoses every run that failed.
#
# **Reference state.** Nothing to build on. The data is downloaded once into `data/cache/` by
# `mth.public_data`; if the download fails you get a labelled synthetic stand-in, and your
# numbers will then differ from the slides'.
#
# **AI level: AI-tutor.** Hints, not answers (`labs/TUTOR.md`).
#
# **Tasks**
# - [ ] 1 · Load the public data, split it and find the reference loss (15')
# - [ ] 2 · Sweep the learning rate for batch gradient descent (20')
# - [ ] 3 · Sweep it for mini-batch stochastic gradient descent (15')
# - [ ] 4 · Sweep it for momentum (15')
# - [ ] 5 · Run on unscaled features and diagnose (15')
# - [ ] 6 · Write the experiment log with a diagnosis per run (15')
# - [ ] 7 · Add warm-up and step decay to the noisy run (10')
# - [ ] ✱ Stretch · Add an adaptive step method and compare its sweep

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

X_raw, y, SOURCE = U.public_data("airfoil")
print("source:", SOURCE)
print("rows:", len(y), "| features:", U.PUBLIC["airfoil"]["columns"][:-1])

# %% [markdown]
# ## 1 · Split, scale and the reference loss (15')
#
# The loss is half the mean squared error, `L(w) = 0.5 * mean((A w - y)**2)`, where `A` holds the
# features and a column of ones for the intercept. Its gradient is `A.T @ (A w - y) / n`.
#
# ✏️ **Exercise** · Split with the fixed permutation below (1,200 rows to train, the rest held
# out), standardise each feature with the **training** mean and standard deviation, add the
# column of ones, and compute the least-squares solution (`np.linalg.lstsq`) and its training
# loss: the best any optimiser can reach.

# %%
rng = np.random.default_rng(5)
perm = rng.permutation(len(y))
TR, TE = perm[:1200], perm[1200:]


def prepare(X):
    # BEGIN SOLUTION
    mu, sd = X[TR].mean(axis=0), X[TR].std(axis=0)
    A = np.column_stack([(X - mu) / sd, np.ones(len(X))])
    return A[TR], y[TR], A[TE], y[TE]
    # END SOLUTION


def loss(A, b, w):
    r = A @ w - b
    return 0.5 * float(np.mean(r * r))


def grad(A, b, w):
    return A.T @ (A @ w - b) / len(b)

# %%
try:
    A_tr, y_tr, A_te, y_te = prepare(X_raw)
except NotImplementedError:
    mu, sd = X_raw[TR].mean(axis=0), X_raw[TR].std(axis=0)
    A_all = np.column_stack([(X_raw - mu) / sd, np.ones(len(y))])
    A_tr, y_tr, A_te, y_te = A_all[TR], y[TR], A_all[TE], y[TE]
    print("(safety net: the course's own split and scaling)")
w_star = np.linalg.lstsq(A_tr, y_tr, rcond=None)[0]
REF = loss(A_tr, y_tr, w_star)
W0 = np.zeros(A_tr.shape[1])
print(f"reference training loss {REF:.3f} | held-out {loss(A_te, y_te, w_star):.3f} | starting loss {loss(A_tr, y_tr, W0):.1f}")
H = A_tr.T @ A_tr / len(y_tr)
eig = np.linalg.eigvalsh(H)
print("curvatures (eigenvalues of A.T A / n):", np.round(eig, 3), "| 2 / largest =", round(2 / eig[-1], 3))

# %% [markdown]
# ## 2 · Batch gradient descent (20')
#
# ✏️ **Exercise** · `batch_gd(lr, epochs)` starts at zero and takes one step on the full
# training set per epoch; it returns the training loss after every epoch (a list, starting with
# the loss at zero). Stop early and return what you have if the loss stops being finite.

# %%
EPOCHS = 200
LRS_BATCH = [0.01, 0.03, 0.1, 0.3, 0.9, 1.0]


def batch_gd(lr, epochs=EPOCHS, A=None, b=None):
    A = A_tr if A is None else A
    b = y_tr if b is None else b
    # BEGIN SOLUTION
    w = np.zeros(A.shape[1])
    hist = [loss(A, b, w)]
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(epochs):
            w = w - lr * grad(A, b, w)
            hist.append(loss(A, b, w))
            if not np.isfinite(hist[-1]):
                break
    return hist
    # END SOLUTION

# %%
def plot_runs(runs, title, name):
    plt = U.plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, hist in runs.items():
        h = np.array(hist, dtype=float)
        h = np.where(np.isfinite(h), h, np.nan)
        ax.semilogy(np.arange(len(h)), h, label=label)
    ax.axhline(REF, color="black", ls=":", lw=1, label="least squares")
    ax.set_ylim(REF * 0.8, 1e5), ax.set_xlabel("epoch"), ax.set_ylabel("training loss")
    ax.set_title(title), ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(U.out(name), dpi=120)
    plt.close(fig)


try:
    batch_runs = {f"lr={lr}": batch_gd(lr) for lr in LRS_BATCH}
except NotImplementedError:
    batch_runs = U.reference("w5_batch")
plot_runs(batch_runs, "batch gradient descent", "W5_1_batch.png")
for k, h in batch_runs.items():
    print(f"batch {k:9} final {h[-1]:12.3f}  epochs run {len(h) - 1}")

# %% [markdown]
# **Look at** the two largest rates: 2 divided by the largest curvature is the edge. Just below
# it the loss still falls, slowly settling; just above it the loss grows without bound.

# %% [markdown]
# ## 3 · Mini-batch stochastic gradient descent (15')
#
# ✏️ **Exercise** · `sgd(lr, batch=32, seed=0)`: each epoch, shuffle the training rows with
# `np.random.default_rng(seed + epoch)`, then step once per mini-batch of 32 rows. Record the
# **full** training loss once per epoch.

# %%
LRS_SGD = [0.003, 0.01, 0.03, 0.1, 0.3]


def sgd(lr, batch=32, seed=0, epochs=EPOCHS, schedule=None):
    # BEGIN SOLUTION
    w = np.zeros(A_tr.shape[1])
    hist = [loss(A_tr, y_tr, w)]
    step = 0
    with np.errstate(over="ignore", invalid="ignore"):
        for epoch in range(epochs):
            order = np.random.default_rng(seed + epoch).permutation(len(y_tr))
            for i in range(0, len(order), batch):
                idx = order[i:i + batch]
                rate = lr if schedule is None else lr * schedule(step)
                w = w - rate * grad(A_tr[idx], y_tr[idx], w)
                step += 1
            hist.append(loss(A_tr, y_tr, w))
            if not np.isfinite(hist[-1]):
                break
    return hist
    # END SOLUTION

# %%
try:
    sgd_runs = {f"lr={lr}": sgd(lr) for lr in LRS_SGD}
except NotImplementedError:
    sgd_runs = U.reference("w5_sgd")
plot_runs(sgd_runs, "mini-batch SGD, batch 32", "W5_1_sgd.png")
for k, h in sgd_runs.items():
    tail = np.array(h[-20:])
    print(f"sgd   {k:9} final {h[-1]:10.3f}  spread of the last 20 epochs {tail.max() - tail.min():8.3f}")

# %% [markdown]
# ## 4 · Momentum (15')
#
# Heavy-ball momentum keeps a velocity: `v = beta * v + g`, then `w = w - lr * v`. With
# `beta = 0.9` it averages about the last ten gradients.
#
# ✏️ **Exercise** · Write `momentum_gd(lr, beta=0.9)`, full batch, as in task 2.

# %%
LRS_MOM = [0.001, 0.01, 0.03, 0.1, 0.3]


def momentum_gd(lr, beta=0.9, epochs=EPOCHS):
    # BEGIN SOLUTION
    w = np.zeros(A_tr.shape[1])
    v = np.zeros_like(w)
    hist = [loss(A_tr, y_tr, w)]
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(epochs):
            v = beta * v + grad(A_tr, y_tr, w)
            w = w - lr * v
            hist.append(loss(A_tr, y_tr, w))
            if not np.isfinite(hist[-1]):
                break
    return hist
    # END SOLUTION

# %%
try:
    mom_runs = {f"lr={lr}": momentum_gd(lr) for lr in LRS_MOM}
except NotImplementedError:
    mom_runs = U.reference("w5_momentum")
plot_runs(mom_runs, "momentum, beta = 0.9", "W5_1_momentum.png")
for k, h in mom_runs.items():
    print(f"mom   {k:9} final {h[-1]:12.3f}  epochs run {len(h) - 1}")

# %% [markdown]
# ## 5 · Unscaled features (15')
#
# The frequency column runs to about 20,000 and the thickness column stays below 0.07. Without
# scaling, the curvatures differ by about eleven orders of magnitude.
#
# ✏️ **Exercise** · Build the unscaled design matrix (raw features and the column of ones),
# compute its largest and smallest curvature, and run batch gradient descent at a learning rate
# of 1e-7 (just stable) and of 1e-6. Return the two loss histories and the two curvatures.

# %%
def unscaled_runs():
    # BEGIN SOLUTION
    A_raw = np.column_stack([X_raw[TR], np.ones(len(TR))])
    e = np.linalg.eigvalsh(A_raw.T @ A_raw / len(TR))
    return {"lr=1e-7": batch_gd(1e-7, A=A_raw), "lr=1e-6": batch_gd(1e-6, A=A_raw)}, float(e[-1]), float(e[0])
    # END SOLUTION

# %%
try:
    raw_runs, lam_max, lam_min = unscaled_runs()
except NotImplementedError:
    ref = U.reference("w5_unscaled")
    raw_runs, lam_max, lam_min = ref["runs"], ref["lam_max"], ref["lam_min"]
print(f"unscaled curvatures: largest {lam_max:.3g}, smallest {lam_min:.3g}, ratio {lam_max / lam_min:.2g}")
for k, h in raw_runs.items():
    print(f"raw   {k:9} final {h[-1]:12.4g}  epochs run {len(h) - 1}")

# %% [markdown]
# ## 6 · The experiment log (15')
#
# ✏️ **Exercise** · `diagnose(hist)` names what a run did, from its loss history alone:
#
# - `"diverged"`: the last loss is not finite, or it is above the starting loss;
# - `"slow"`: the final loss is more than 10% above the reference;
# - `"noisy"`: close to the reference, but the last 20 epochs still move by more than 5% of the
#   reference;
# - `"converged"`: none of the above.
#
# The cell writes the log of every run of tasks 2 to 5 to `out/W5_1_experiments.csv`.

# %%
def diagnose(hist):
    # BEGIN SOLUTION
    h = np.array(hist, dtype=float)
    if not np.isfinite(h[-1]) or h[-1] > h[0]:
        return "diverged"
    if h[-1] > 1.1 * REF:
        return "slow"
    if h[-20:].max() - h[-20:].min() > 0.05 * REF:
        return "noisy"
    return "converged"
    # END SOLUTION

# %%
rows = []
for variant, runs in (("batch", batch_runs), ("sgd", sgd_runs), ("momentum", mom_runs), ("batch, unscaled", raw_runs)):
    for k, h in runs.items():
        try:
            d = diagnose(h)
        except NotImplementedError:
            d = U.reference("w5_diagnoses")[f"{variant} {k}"]
        rows.append((variant, k[3:], len(h) - 1, h[-1], d))
with open(U.out("W5_1_experiments.csv"), "w", encoding="utf-8") as fh:
    fh.write("variant,learning_rate,epochs_run,final_training_loss,diagnosis\n")
    for r in rows:
        fh.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.4g},{r[4]}\n")
for r in rows:
    print(f"{r[0]:16} lr={r[1]:7} epochs={r[2]:3}  final={r[3]:11.4g}  {r[4]}")

# %% [markdown]
# **Look at** every row that is not "converged" and write, in the log, the cause in one line:
# a rate past 2 / largest curvature, a rate too small for the smallest curvature, or the noise of
# small batches at a large rate.

# %% [markdown]
# ## 7 · Warm-up and step decay (10')
#
# The noisiest stable SGD run keeps bouncing around the minimum because its rate never shrinks.
#
# ✏️ **Exercise** · Write `schedule(step)`: a linear warm-up from 0.1 to 1 over the first 100
# steps, then the rate is halved every 1,500 steps. Run `sgd(0.3, schedule=schedule)`, the noisy
# rate of task 3, and compare the spread of the last 20 epochs with the constant-rate run.

# %%
def schedule(step):
    # BEGIN SOLUTION
    if step < 100:
        return 0.1 + 0.9 * step / 100
    return 0.5 ** ((step - 100) // 1500)
    # END SOLUTION

# %%
try:
    sched = sgd(0.3, schedule=schedule)
except NotImplementedError:
    sched = U.reference("w5_schedule")
const = sgd_runs["lr=0.3"]
spread = lambda h: float(np.max(h[-20:]) - np.min(h[-20:]))  # noqa: E731
print(f"lr=0.3 constant:          final {const[-1]:.3f}, spread of the last 20 epochs {spread(const):.3f}")
print(f"lr=0.3 warm-up and decay: final {sched[-1]:.3f}, spread of the last 20 epochs {spread(sched):.3f}")
plot_runs({"constant 0.3": const, "warm-up and decay": sched}, "SGD with a schedule", "W5_1_schedule.png")

# %% [markdown]
# ## ✱ Stretch · An adaptive step method
#
# Adam scales each coordinate's step by a running estimate of its gradient's size. Write it (the
# update is in the week 5 slides), run the same sweep as task 3 with mini-batches, and compare which
# rates converge.

# %%
def adam(lr, b1=0.9, b2=0.999, eps=1e-8, batch=32, seed=0, epochs=EPOCHS):
    w = np.zeros(A_tr.shape[1])
    m, v, t = np.zeros_like(w), np.zeros_like(w), 0
    hist = [loss(A_tr, y_tr, w)]
    for epoch in range(epochs):
        order = np.random.default_rng(seed + epoch).permutation(len(y_tr))
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            g = grad(A_tr[idx], y_tr[idx], w)
            t += 1
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            w = w - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        hist.append(loss(A_tr, y_tr, w))
    return hist


for lr in (0.01, 0.1, 1.0):
    h = adam(lr)
    print(f"adam lr={lr:5}: final {h[-1]:.3f}")

# %% [markdown]
# ## Checkpoint questions
#
# 1. The largest curvature of a quadratic loss is 4. Which learning rates make batch gradient
#    descent diverge?
#    <details><summary>Answer</summary>Rates above 2 / 4 = 0.5.</details>
# 2. A loss curve falls fast for ten epochs, then crawls far above the reference. Name a cause and
#    a fix.
#    <details><summary>Answer</summary>Ill-conditioning: the rate fits the steepest direction
#    and is too small for the flattest; scale the features (or use momentum).</details>
# 3. Why does a constant-rate SGD run never settle exactly at the minimum?
#    <details><summary>Answer</summary>Each mini-batch gradient is noisy; with a fixed rate the
#    noise keeps moving the weights. A decaying rate shrinks the moves.</details>
#
# **Carry forward.** The experiment log `out/W5_1_experiments.csv` is the model for the
# capstone's log: every run, its rate, its seed and its diagnosis.
