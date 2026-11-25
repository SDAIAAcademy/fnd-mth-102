# %% [markdown]
# # Lab 3.1 · Backpropagation by hand
#
# Week 3 · Session 1 (Graphs and the chain rule) · 105 min.
#
# **AI level: AI-off.** No assistant, no tutor: this lab is the practice for the supervised tests
# and the oral check. Paper, pencil and a calculator. The notebook only checks the numbers you
# worked out by hand.
#
# The charity's completion model is a logistic regression. You take two learners, LR-0007 and
# LR-0042, with two scaled features each, and the loss averaged over the two. You draw its
# computational graph, run the forward pass and the backward pass by hand, then check your
# gradient against code and against a numerical gradient check.
#
# **Reference state.** Nothing to build on: the two learners, the weights and the bias are in
# `data/case.json` (synthetic).
#
# **Tasks**
# - [ ] 1 · Draw the graph of the two-example loss (15')
# - [ ] 2 · Fill in the forward pass by hand (15')
# - [ ] 3 · Run the backward pass by hand (25')
# - [ ] 4 · Check your forward pass in code (10')
# - [ ] 5 · Match a numerical gradient check (15')
# - [ ] 6 · One directional derivative in forward mode (15')
# - [ ] 7 · Write where your values differed, and why (10')
# - [ ] ✱ Stretch · Add an L2 term and extend the backward pass by hand

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
import numpy as np
import mth as U

T = U.case()["two_learners"]
X = np.array(T["x"])
Y = np.array(T["y"])
W = np.array(T["w"])
B = T["b"]
for code, x, y in zip(T["codes"], X, Y):
    print(f"{code}: x = {x.tolist()}, completed = {y}")
print("weights w =", W.tolist(), " bias b =", B)

# %% [markdown]
# ## The loss
#
# For learner `i`: `z_i = w1*x_i1 + w2*x_i2 + b`, `p_i = sigmoid(z_i)`,
# `l_i = -(y_i * log(p_i) + (1 - y_i) * log(1 - p_i))`. The loss is `L = (l_1 + l_2) / 2`.
#
# ## 1 · The graph of the two-example loss (15')
#
# Draw it on paper first: one node per operation, inputs at the bottom, `L` at the top.
#
# ✏️ **Exercise** · Write the graph as a list of nodes in an order where every node comes after
# its inputs. Each node is `(name, op, inputs)` with `op` one of `"lin"` (the linear score of
# one learner: inputs are the learner's index as a string, `"w1"`, `"w2"`, `"b"`), `"sigmoid"`,
# `"bce"` (the cross-entropy of one learner: inputs are its `p` and the learner's index) and
# `"mean"`. The cell below evaluates your graph.

# %%
def graph():
    # BEGIN SOLUTION
    return [
        ("z1", "lin", ["0", "w1", "w2", "b"]),
        ("p1", "sigmoid", ["z1"]),
        ("l1", "bce", ["p1", "0"]),
        ("z2", "lin", ["1", "w1", "w2", "b"]),
        ("p2", "sigmoid", ["z2"]),
        ("l2", "bce", ["p2", "1"]),
        ("L", "mean", ["l1", "l2"]),
    ]
    # END SOLUTION


def evaluate(nodes, w, b):
    v = {"w1": w[0], "w2": w[1], "b": b}
    for name, op, ins in nodes:
        if op == "lin":
            i = int(ins[0])
            v[name] = v["w1"] * X[i, 0] + v["w2"] * X[i, 1] + v["b"]
        elif op == "sigmoid":
            v[name] = 1 / (1 + math.exp(-v[ins[0]]))
        elif op == "bce":
            p, y = v[ins[0]], Y[int(ins[1])]
            v[name] = -(y * math.log(p) + (1 - y) * math.log(1 - p))
        elif op == "mean":
            v[name] = sum(v[k] for k in ins) / len(ins)
    return v

# %%
try:
    NODES = graph()
except NotImplementedError:
    NODES = [(n, o, i) for n, o, i in U.reference("w3_graph")]
VALUES = evaluate(NODES, W, B)
print(f"{len(NODES)} nodes; the graph gives L = {VALUES['L']:.4f}")

# %% [markdown]
# ## 2 · The forward pass by hand (15')
#
# ✏️ **Exercise** · Work out every node's value on paper, to four decimals, and type them in.
# Do not copy them from the cell above: that is what task 4 checks against.

# %%
def hand_forward():
    # BEGIN SOLUTION
    return {"z1": 0.65, "p1": 0.6570, "l1": 0.4201, "z2": 0.70, "p2": 0.6682, "l2": 1.1032, "L": 0.7616}
    # END SOLUTION

# %% [markdown]
# ## 3 · The backward pass by hand (25')
#
# The adjoint of a node is `dL/d(node)`. Start at `L` (adjoint 1) and go down the graph: each
# node passes its adjoint, times its local derivative, to each of its inputs. Where a node feeds
# two others, its adjoint is the **sum** of what it receives.
#
# Local derivatives you need: `d mean / d l_i = 1/2`; `d bce / d p = -y/p + (1-y)/(1-p)`;
# `d sigmoid / d z = p (1 - p)`; `d z_i / d w_j = x_ij`, `d z_i / d b = 1`.
#
# ✏️ **Exercise** · Type in the adjoint of every node, to four decimals.

# %%
def hand_backward():
    # BEGIN SOLUTION
    return {"L": 1.0, "l1": 0.5, "l2": 0.5, "p1": -0.7610, "p2": 1.5069, "z1": -0.1715, "z2": 0.3341,
            "w1": -0.0207, "w2": -0.0361, "b": 0.1626}
    # END SOLUTION

# %%
try:
    fwd = hand_forward()
    bwd = hand_backward()
except NotImplementedError:
    ref = U.reference("w3_hand")
    fwd, bwd = ref["forward"], ref["backward"]
print(f"{'node':5} {'your forward':>13} {'your adjoint':>13}")
for k in ["z1", "p1", "l1", "z2", "p2", "l2", "L"]:
    print(f"{k:5} {fwd[k]:13.4f} {bwd[k]:13.4f}")
print("gradient by hand: dL/dw1 = %.4f, dL/dw2 = %.4f, dL/db = %.4f" % (bwd["w1"], bwd["w2"], bwd["b"]))

# %% [markdown]
# ## 4 · Check your forward pass in code (10')
#
# ✏️ **Exercise** · Write `loss(w, b)` with NumPy, vectorised over the two learners. Then the
# cell compares every one of your hand values with the code's.

# %%
def loss(w, b):
    # BEGIN SOLUTION
    p = 1 / (1 + np.exp(-(X @ w + b)))
    return float(np.mean(-(Y * np.log(p) + (1 - Y) * np.log(1 - p))))
    # END SOLUTION

# %%
try:
    L_code = loss(W, B)
except NotImplementedError:
    L_code = VALUES["L"]
    loss = lambda w, b: evaluate(NODES, w, b)["L"]  # noqa: E731
print(f"L from code: {L_code:.6f}")
off = {k: abs(fwd[k] - VALUES[k]) for k in fwd}
for k, d in off.items():
    print(f"{k:3} hand {fwd[k]:.4f}  code {VALUES[k]:.4f}  {'ok' if d < 5e-4 else 'CHECK THIS ONE'}")
assert all(d < 5e-4 for d in off.values()), "a hand value is off by more than rounding"

# %% [markdown]
# ## 5 · A numerical gradient check (15')
#
# ✏️ **Exercise** · Compute the gradient of `loss` with respect to `(w1, w2, b)` by central
# differences and return the relative error of your hand gradient against it.

# %%
def check_hand_gradient():
    # BEGIN SOLUTION
    f = lambda th: loss(th[:2], th[2])  # noqa: E731
    g_num = U.central_diff(f, np.array([W[0], W[1], B]))
    g_hand = np.array([bwd["w1"], bwd["w2"], bwd["b"]])
    return g_num, U.rel_err(g_hand, g_num)
    # END SOLUTION

# %%
try:
    g_num, err = check_hand_gradient()
except NotImplementedError:
    g_num = U.central_diff(lambda th: loss(th[:2], th[2]), np.array([W[0], W[1], B]))
    err = U.rel_err(np.array([bwd["w1"], bwd["w2"], bwd["b"]]), g_num)
print("numerical gradient:", np.round(g_num, 6))
print(f"relative error of your hand gradient: {err:.1e}  ({'within hand rounding' if err < 5e-3 else 'look again'})")
assert err < 5e-3

# %% [markdown]
# **Look at** the adjoint of `z_i`: it is `(p_i - y_i) / 2`. The sigmoid's derivative and the
# cross-entropy's derivative cancel into the prediction error. That is why this pair is used
# together.

# %% [markdown]
# ## 6 · Forward mode: one directional derivative (15')
#
# Forward mode carries a tangent `dot(node) = d node / d w1` up the graph alongside each value:
# `dot(w1) = 1`, `dot(w2) = dot(b) = 0`. Each node's tangent is the sum of its local derivatives
# times its inputs' tangents.
#
# ✏️ **Exercise** · Work out the tangents by hand and type them in. Then answer: how many forward
# passes give the whole gradient of `L`, and how many backward passes?

# %%
def hand_tangents():
    # BEGIN SOLUTION
    return {"z1": 0.9, "p1": 0.2028, "l1": -0.3087, "z2": 0.4, "p2": 0.0887, "l2": 0.2673, "L": -0.0207}
    # END SOLUTION


def passes_needed():
    # BEGIN SOLUTION
    return {"forward": 3, "backward": 1}
    # END SOLUTION

# %%
try:
    tang, passes = hand_tangents(), passes_needed()
except NotImplementedError:
    ref = U.reference("w3_forward_mode")
    tang, passes = ref["tangents"], ref["passes"]
print(f"forward mode dL/dw1 = {tang['L']:.4f}; reverse mode gave {bwd['w1']:.4f}; numerical {g_num[0]:.4f}")
print("passes for the full gradient:", passes)
assert abs(tang["L"] - g_num[0]) < 5e-4 and passes == {"forward": 3, "backward": 1}

# %% [markdown]
# ## 7 · Where your values differed, and why (10')
#
# ✏️ **Exercise** · Write where your hand values first differed from the code, the cause (a sign,
# a missed sum, rounding), and what you will check first next time.

# %%
def findings():
    # BEGIN SOLUTION
    return [
        f"Hand gradient (dL/dw1, dL/dw2, dL/db) = ({bwd['w1']:.4f}, {bwd['w2']:.4f}, {bwd['b']:.4f}); "
        f"relative error against central differences {err:.0e}, which is four-decimal rounding.",
        "The adjoint of each z is (p - y) / 2: negative for LR-0007 (completed), so a descent step raises its "
        "score; positive for LR-0042 (did not), so a step lowers its score. The gradient adds the two pulls.",
        "Forward mode needs one pass per parameter (three here); reverse mode one backward pass for all three.",
    ]
    # END SOLUTION

# %%
try:
    log = findings()
except NotImplementedError:
    log = ["(write your findings here)"]
U.out("W3_1_log.md").write_text("# Lab 3.1 log\n\n" + "\n".join(f"- {s}" for s in log) + "\n", encoding="utf-8")
print("\n".join(log))

# %% [markdown]
# ## ✱ Stretch · An L2 term
#
# Add `lam/2 * (w1**2 + w2**2)` to `L`, with `lam = 0.1`. Which adjoints change? Work out the new
# gradient by hand, then check it here with `U.central_diff`.

# %%
lam = 0.1
f_l2 = lambda th: loss(th[:2], th[2]) + lam / 2 * (th[0] ** 2 + th[1] ** 2)  # noqa: E731
print("numerical gradient with the L2 term:", np.round(U.central_diff(f_l2, np.array([W[0], W[1], B])), 4))

# %% [markdown]
# ## Checkpoint questions
#
# 1. A node feeds two later nodes. How is its adjoint formed?
#    <details><summary>Answer</summary>As the sum of the two contributions, one from each path:
#    the multivariable chain rule.</details>
# 2. Why does the adjoint of `z` come out as `p - y` for one learner?
#    <details><summary>Answer</summary>The cross-entropy's derivative with respect to p,
#    -y/p + (1-y)/(1-p), times the sigmoid's p(1-p), simplifies to p - y.</details>
# 3. A model has a million weights and one loss. Which mode gives the gradient in fewer passes?
#    <details><summary>Answer</summary>Reverse mode: one backward pass, against a million
#    forward passes.</details>
#
# **Carry forward.** In week 4 you write an engine that does task 3 for any graph. Keep your hand
# values: the engine's first test compares with them. The log is in `out/W3_1_log.md`.
