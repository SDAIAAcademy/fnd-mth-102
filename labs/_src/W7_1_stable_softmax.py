# %% [markdown]
# # Lab 7.1 · Stable softmax and a penalty
#
# Week 7 · Session 1 (Stable losses and penalties) · 105 min.
#
# The charity places each new learner in one of three class levels (0, 1, 2). You watch a naive
# softmax overflow, write a stable one with its cross-entropy, check its gradient, train a
# three-level placement model with gradient descent, then add an L2 penalty and measure what it
# changes: the weights, the training loss and the held-out loss.
#
# **Reference state.** Nothing to build on: the data are in `data/learners.csv` (synthetic).
#
# **AI level: AI-tutor.** Hints, not answers (`labs/TUTOR.md`).
#
# **Tasks**
# - [ ] 1 · Watch naive softmax overflow on the planted logits (10')
# - [ ] 2 · Write stable log-sum-exp, softmax and cross-entropy (20')
# - [ ] 3 · Check the gradient of softmax cross-entropy (15')
# - [ ] 4 · Train the placement model with gradient descent (20')
# - [ ] 5 · Add an L2 penalty and sweep its strength (15')
# - [ ] 6 · Compare weight norms and held-out loss (15')
# - [ ] 7 · Write three findings in the lab log (10')
# - [ ] ✱ Stretch · Show that the penalised loss is strictly convex while the plain one is not

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
L = U.learners()
STEPS = U.pick(2000, 300)
print("levels 0, 1, 2:", np.bincount(L["level"]).tolist(), " steps per training run:", STEPS)

# %% [markdown]
# ## 1 · Naive softmax overflows (10')
#
# `softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)`. A badly scaled model produced the three rows of logits in
# `case.json` (`overflow_logits`). The cell computes the naive softmax and its cross-entropy for
# the labels 1, 2 and 0. Before running it, predict what each row prints.

# %%
Z_BAD = np.array(C["overflow_logits"])
Y_BAD = np.array([1, 2, 0])
with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
    E = np.exp(Z_BAD)
    P_naive = E / E.sum(axis=1, keepdims=True)
    ce_naive = -np.log(P_naive[np.arange(3), Y_BAD])
for z, p, c in zip(Z_BAD, P_naive, ce_naive):
    print(f"logits {z}: naive softmax {p}, cross-entropy {c}")

# %% [markdown]
# **Look at** the two ways it fails. In the first two rows `exp(812)` and `exp(820)` are `inf`,
# and `inf / inf` is `nan`. In the last row the `exp` values are so small that they keep only a
# few significant digits, and the probability of the label rounds to exactly 0, so its
# cross-entropy is `inf`. Nothing is wrong with the mathematics: softmax does not change when the
# same number is subtracted from every logit of a row.
#
# ## 2 · Stable log-sum-exp, softmax and cross-entropy (20')
#
# ✏️ **Exercise** · For a matrix `Z` (one row per learner), write
#
# - `logsumexp(Z)`: `m + log Σ exp(z − m)` per row, with `m` the row's largest logit;
# - `softmax(Z)`: `exp(z − logsumexp)`, per row;
# - `cross_entropy(Z, y)`: the mean over rows of `logsumexp(z) − z_y`, never taking `log` of a
#   probability.

# %%
def logsumexp(Z):
    # BEGIN SOLUTION
    m = Z.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(Z - m).sum(axis=1, keepdims=True)))[:, 0]
    # END SOLUTION


def softmax(Z):
    # BEGIN SOLUTION
    return np.exp(Z - logsumexp(Z)[:, None])
    # END SOLUTION


def cross_entropy(Z, y):
    # BEGIN SOLUTION
    return float(np.mean(logsumexp(Z) - Z[np.arange(len(y)), y]))
    # END SOLUTION

# %%
try:
    stable = {"softmax": np.round(softmax(Z_BAD), 6).tolist(), "cross_entropy": cross_entropy(Z_BAD, Y_BAD)}
except NotImplementedError:
    stable = U.reference("w7_stable")
for row in stable["softmax"]:
    print("stable softmax", row)
print(f"mean cross-entropy {stable['cross_entropy']:.4f}")

# %% [markdown]
# ## 3 · The gradient of softmax cross-entropy (15')
#
# For one row, the derivative of `logsumexp(z) − z_y` with respect to the logits is `p − e_y`:
# the predicted probabilities minus the one-hot label. For the mean over `n` rows, divide by `n`.
# With logits `Z = X W`, the chain rule gives `Xᵀ (P − Y) / n` for the weights.
#
# ✏️ **Exercise** · `grad_check()` compares both formulas with `U.central_diff` on the small
# example below and returns the two relative errors.

# %%
rng = np.random.default_rng(7)
Z_small = rng.normal(0, 2, (5, 3))
y_small = np.array([0, 2, 1, 1, 0])
X_small = rng.normal(0, 1, (5, 4))
W_small = rng.normal(0, 0.5, (4, 3))


def grad_check():
    # BEGIN SOLUTION
    Y = np.eye(3)[y_small]
    gZ = (softmax(Z_small) - Y) / len(y_small)
    nZ = U.central_diff(lambda z: cross_entropy(z.reshape(5, 3), y_small), Z_small.ravel()).reshape(5, 3)
    gW = X_small.T @ (softmax(X_small @ W_small) - Y) / len(y_small)
    nW = U.central_diff(lambda w: cross_entropy(X_small @ w.reshape(4, 3), y_small), W_small.ravel()).reshape(4, 3)
    return U.rel_err(gZ, nZ), U.rel_err(gW, nW)
    # END SOLUTION

# %%
try:
    errs = grad_check()
except NotImplementedError:
    errs = U.reference("w7_gradcheck")
print(f"relative error, logits: {errs[0]:.1e}; weights: {errs[1]:.1e}")

# %% [markdown]
# ## 4 · Train the placement model (20')
#
# Three features (entry score, attendance, weekly hours), standardised on the training rows, and
# a bias column: `X` has four columns and `W` is 4 × 3. The split is fixed: 450 learners to train,
# 150 held out.
#
# ✏️ **Exercise** · `train(lam, steps, lr=0.5)` starts from `W = 0` and takes `steps` steps of
# gradient descent on `cross_entropy(X W, y) + (lam / 2) ‖W‖²`. The penalty covers all twelve
# parameters, bias included. It returns `W` and a history: a list of `(step, data loss, ‖W‖)`
# every 50 steps, where the data loss is the cross-entropy without the penalty.

# %%
X_raw = np.column_stack([L["entry_score"], L["attendance"], L["weekly_hours"]]).astype(float)
y_all = L["level"]
perm = np.random.default_rng(102).permutation(len(y_all))
TR, TE = perm[:450], perm[450:]
mu, sd = X_raw[TR].mean(axis=0), X_raw[TR].std(axis=0)
X_all = np.column_stack([(X_raw - mu) / sd, np.ones(len(y_all))])
Xtr, ytr, Xte, yte = X_all[TR], y_all[TR], X_all[TE], y_all[TE]


def train(lam, steps, lr=0.5):
    # BEGIN SOLUTION
    W = np.zeros((4, 3))
    Y = np.eye(3)[ytr]
    hist = []
    for t in range(1, steps + 1):
        G = Xtr.T @ (softmax(Xtr @ W) - Y) / len(ytr) + lam * W
        W = W - lr * G
        if t % 50 == 0:
            hist.append((t, cross_entropy(Xtr @ W, ytr), float(np.linalg.norm(W))))
    return W, hist
    # END SOLUTION

# %%
try:
    W_plain, hist_plain = train(0.0, STEPS)
    plain = {"train": cross_entropy(Xtr @ W_plain, ytr), "held_out": cross_entropy(Xte @ W_plain, yte),
             "accuracy": float(np.mean((Xte @ W_plain).argmax(axis=1) == yte)), "history": hist_plain}
except NotImplementedError:
    plain = U.reference("w7_plain")
for t, loss, norm in plain["history"][:: max(1, len(plain["history"]) // 8)] + plain["history"][-1:]:
    print(f"step {t:5}: training loss {loss:.4f}, weight norm {norm:.3f}")
print(f"held-out loss {plain['held_out']:.4f}, held-out accuracy {plain['accuracy']:.3f}")

# %% [markdown]
# **Look at** the weight norm: it is still growing at the last step. The levels are set by
# thresholds on the entry score alone, so the training data can be separated perfectly. The loss
# can always fall a little further by making the weights larger: the plain loss has no finite
# minimiser, and gradient descent never stops on its own.
#
# ## 5 · An L2 penalty (15')
#
# ✏️ **Exercise** · `sweep()` trains with `lam` in `LAMS` (same steps, same rate) and returns one
# row per strength: `lam`, final ‖W‖, training data loss, held-out loss, held-out accuracy.

# %%
LAMS = [0.0, 1e-3, 1e-2, 1e-1, 1.0]


def sweep():
    # BEGIN SOLUTION
    rows = []
    for lam in LAMS:
        W, _ = train(lam, STEPS)
        rows.append([lam, float(np.linalg.norm(W)), cross_entropy(Xtr @ W, ytr), cross_entropy(Xte @ W, yte),
                     float(np.mean((Xte @ W).argmax(axis=1) == yte))])
    return rows
    # END SOLUTION

# %%
try:
    rows = sweep()
except NotImplementedError:
    rows = U.reference("w7_sweep")
print(f"{'lam':>6} {'|W|':>8} {'train':>8} {'held-out':>9} {'accuracy':>9}")
for lam, norm, tr_loss, te_loss, acc in rows:
    print(f"{lam:6g} {norm:8.3f} {tr_loss:8.4f} {te_loss:9.4f} {acc:9.3f}")

# %% [markdown]
# ## 6 · Weight norms and held-out loss (15')
#
# The cell plots the weight norm over the steps for the plain model and for `lam = 0.01`, and the
# held-out loss against the strength.
#
# ✏️ **Exercise** · Answer in `compare()`: which strength gave the lowest held-out loss, and did
# any penalty improve on the plain model here? Return the strength and `True` or `False`.

# %%
plt = U.plt()
try:
    _, hist_pen = train(1e-2, STEPS)
except NotImplementedError:
    hist_pen = U.reference("w7_penalised_history")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.plot([h[0] for h in plain["history"]], [h[2] for h in plain["history"]], label="plain")
a1.plot([h[0] for h in hist_pen], [h[2] for h in hist_pen], label="lam = 0.01")
a1.set_xlabel("step"), a1.set_ylabel("weight norm"), a1.legend()
a2.semilogx([max(r[0], 1e-4) for r in rows], [r[3] for r in rows], "o-")
a2.set_xlabel("lam (plain shown at 1e-4)"), a2.set_ylabel("held-out loss")
fig.tight_layout()
fig.savefig(U.out("W7_1_penalty.png"), dpi=110)
plt.close(fig)
print("saved out/W7_1_penalty.png")


def compare():
    # BEGIN SOLUTION
    best = min(rows, key=lambda r: r[3])
    return best[0], bool(any(r[3] < rows[0][3] for r in rows[1:]))
    # END SOLUTION

# %%
try:
    best_lam, penalty_helped = compare()
except NotImplementedError:
    best_lam, penalty_helped = U.reference("w7_compare")
print(f"lowest held-out loss at lam = {best_lam:g}; a penalty improved on the plain model: {penalty_helped}")

# %% [markdown]
# **Look at** the price and the gain. These labels carry no noise, so the plain model's
# overconfidence costs nothing on held-out learners of the same kind, and every penalty raised
# the held-out loss. What the penalty buys is a finite, unique solution with bounded weights: a
# model whose training stops somewhere. On noisy labels the trade can come out the other way;
# measure it, do not assume it.
#
# ## 7 · The lab log (10')
#
# ✏️ **Exercise** · Three findings with numbers: the overflow and its fix, the gradient check, and
# what the penalty changed.

# %%
def findings():
    # BEGIN SOLUTION
    norm_first, norm_last = plain["history"][0][2], plain["history"][-1][2]
    r1 = [r for r in rows if r[0] == 1e-2][0]
    return [
        "Naive cross-entropy was nan, nan and inf on the three planted rows (exp overflowed to inf, or the "
        f"label's probability rounded to 0); the shifted version gave finite probabilities and a mean cross-entropy of {stable['cross_entropy']:.2f}.",
        f"The gradient P - Y matched central differences to {errs[0]:.0e} on the logits and {errs[1]:.0e} on the weights.",
        f"Without a penalty the weight norm grew from {norm_first:.2f} at step 50 to {norm_last:.2f} at step "
        f"{plain['history'][-1][0]} and was still growing; at lam = 0.01 it stayed at {r1[1]:.2f}. "
        f"Held-out loss rose from {rows[0][3]:.3f} to {r1[3]:.3f}: on noise-free labels the penalty bought "
        "bounded weights, not a better held-out loss.",
    ]
    # END SOLUTION

# %%
try:
    log = findings()
except NotImplementedError:
    log = ["(write your three findings here)"]
U.out("W7_1_log.md").write_text("# Lab 7.1 log\n\n" + "\n".join(f"- {s}" for s in log) + "\n", encoding="utf-8")
print("\n".join(log))

# %% [markdown]
# ## ✱ Stretch · Strict convexity
#
# Adding the same vector to all three columns of `W` adds the same number to every logit of a
# row, which softmax ignores: the plain loss is flat along that direction, so its Hessian has a
# zero eigenvalue and the loss is convex but not strictly convex. The penalty adds `lam` to every
# eigenvalue. The cell builds the 12 × 12 Hessian from differences of the gradient.

# %%
def hessian_of(lam, W):
    Y = np.eye(3)[ytr]
    g = lambda w: (Xtr.T @ (softmax(Xtr @ w.reshape(4, 3)) - Y) / len(ytr) + lam * w.reshape(4, 3)).ravel()  # noqa: E731
    h, n = 1e-5, W.size
    H = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        H[:, j] = (g(W.ravel() + e) - g(W.ravel() - e)) / (2 * h)
    return (H + H.T) / 2


try:
    W_mid = train(1e-2, 200)[0]
    for lam in (0.0, 1e-2):
        ev = np.linalg.eigvalsh(hessian_of(lam, W_mid))
        print(f"lam = {lam:g}: smallest Hessian eigenvalue {ev[0]:.2e}, largest {ev[-1]:.3f}")
except NotImplementedError:
    print("(finish tasks 2 and 4 first)")

# %% [markdown]
# ## Checkpoint questions
#
# 1. A row of logits is (1000, 1001, 999). What is its softmax, to three decimals?
#    <details><summary>Answer</summary>The same as for (1, 2, 0): about (0.245, 0.665, 0.090).
#    </details>
# 2. The gradient of mean cross-entropy with respect to one row of logits is (0.3, −0.5, 0.2)
#    for n = 1. What was the label, and what probability did the model give it?
#    <details><summary>Answer</summary>Label 1 (the only negative entry); probability
#    1 − 0.5 = 0.5.</details>
# 3. Why does gradient descent on separable data never stop on its own without a penalty?
#    <details><summary>Answer</summary>Scaling the weights up always lowers the loss a little
#    more, so there is no finite minimiser to stop at.</details>
#
# **Carry forward.** In the capstone you train a classifier with your own engine: compute its
# loss from logits with the stable form, never as the log of a probability. Week 8 turns from
# penalties to hard constraints.
