"""Write the reference answers of the labs' safety nets into labs/data/reference/.

    cd labs
    uv run python _tools/make_reference.py

Computed from the case data and the mathematics, without running the labs. Floats are kept to
eight significant digits, so no file holds a long run of digits.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

LABS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABS))
import mth as U  # noqa: E402

REF = LABS / "data" / "reference"


def r(v):
    """Round floats (also inside lists and dicts) to eight significant digits."""
    if isinstance(v, dict):
        return {k: r(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [r(x) for x in v]
    if isinstance(v, (float, np.floating)):
        return float(f"{float(v):.8g}")
    if isinstance(v, np.integer):
        return int(v)
    return v


def save(name: str, value) -> None:
    REF.mkdir(parents=True, exist_ok=True)
    text = json.dumps(r(value), indent=1)
    # large losses of diverging runs in scientific notation, so no long digit run lands in a file
    text = re.sub(r"(?<![\w.])(-?\d{7,})\.0(?!\d)", lambda m: f"{float(m.group(1)):.4e}", text)
    (REF / f"{name}.json").write_text(text + "\n", encoding="utf-8", newline="\n")
    print("wrote reference", name)


def week1(C):
    M, D1 = C["completion_model"], C["districts"]["D1"]
    p = lambda a: U.sigmoid(M["w_a"] * a + M["w_h"] * 4.0 + M["b"])  # noqa: E731
    E = lambda x: D1["k"] * np.log1p(x / D1["s"])  # noqa: E731
    q = lambda t: 12.0 * t * np.exp(-0.25 * t)  # noqa: E731
    s = p(0.6)
    hand = {"p": M["w_a"] * s * (1 - s), "E": D1["k"] / (D1["s"] + 20.0), "q": 12.0 * math.exp(-0.75) * (1 - 0.75)}
    save("w1_hand", hand)
    funcs = {"p": (p, 0.6), "E": (E, 20.0), "q": (q, 3.0)}
    HS = np.logspace(-1, -12, 23)
    sweep = {n: {"forward": [abs((f(x + h) - f(x)) / h - hand[n]) for h in HS],
                 "central": [abs((f(x + h) - f(x - h)) / (2 * h) - hand[n]) for h in HS]} for n, (f, x) in funcs.items()}
    save("w1_sweep", sweep)
    eps = np.finfo(float).eps
    meas = {k: float(np.exp(np.mean([np.log(HS[int(np.argmin(sweep[n][k]))]) for n in funcs]))) for k in ("forward", "central")}
    save("w1_steps", {"theory_forward": math.sqrt(eps), "theory_central": eps ** (1 / 3),
                      "measured_forward": meas["forward"], "measured_central": meas["central"]})
    X0, DXS = 20.0, [1, 2, 5, 10, 15, 20, 30, 40, 60]
    d1, d2 = D1["k"] / (D1["s"] + X0), -D1["k"] / (D1["s"] + X0) ** 2
    T1 = lambda x: E(X0) + d1 * (x - X0)  # noqa: E731
    T2 = lambda x: T1(x) + 0.5 * d2 * (x - X0) ** 2  # noqa: E731
    reach = lambda T: max([dx for dx in DXS if abs(T(X0 + dx) - E(X0 + dx)) / E(X0 + dx) < 0.01] or [0])  # noqa: E731
    save("w1_reach", {"tangent": reach(T1), "quadratic": reach(T2)})


def week2(C):
    M, dist = C["completion_model"], C["districts"]
    K = np.array([dist[d]["k"] for d in dist])
    S = np.array([dist[d]["s"] for d in dist])
    A0, H0 = 0.7, 5.0
    s = U.sigmoid(M["w_a"] * A0 + M["w_h"] * H0 + M["b"])
    g = np.array([M["w_a"] * s * (1 - s), M["w_h"] * s * (1 - s)])
    save("w2_partials", g)
    deg = int(round(math.degrees(math.atan2(g[1], g[0])))) % 360
    save("w2_steepest", [deg, float(np.linalg.norm(g))])
    save("w2_check", 0.0)
    X0 = np.array([30.0, 25.0, 25.0, 20.0, 20.0])
    JE = np.diag(K / (S + X0))
    JF = np.array([[g[0], g[1]], [12 * H0, 12 * A0]])
    save("w2_jacobians", {"JE": JE, "JF": JF})
    E = lambda x: K * np.log1p(x / S)  # noqa: E731
    shifts = {}
    for amount in (5.0, 20.0):
        d = np.array([-amount, 0, 0, amount, 0])
        shifts[str(amount)] = {"predicted": float((JE @ d).sum()), "actual": float(E(X0 + d).sum() - E(X0).sum())}
    save("w2_shifts", shifts)
    save("w2_wrong", [1, 0.0])


def week3(C):
    T = C["two_learners"]
    X, Y, W, B = np.array(T["x"]), np.array(T["y"]), np.array(T["w"]), T["b"]
    save("w3_graph", [["z1", "lin", ["0", "w1", "w2", "b"]], ["p1", "sigmoid", ["z1"]], ["l1", "bce", ["p1", "0"]],
                      ["z2", "lin", ["1", "w1", "w2", "b"]], ["p2", "sigmoid", ["z2"]], ["l2", "bce", ["p2", "1"]],
                      ["L", "mean", ["l1", "l2"]]])
    z = X @ W + B
    p = U.sigmoid(z)
    l_ = -(Y * np.log(p) + (1 - Y) * np.log(1 - p))
    ap = 0.5 * (-Y / p + (1 - Y) / (1 - p))
    az = ap * p * (1 - p)
    aw = X.T @ az
    q = lambda v: round(float(v), 4)  # noqa: E731
    fwd = {"z1": q(z[0]), "p1": q(p[0]), "l1": q(l_[0]), "z2": q(z[1]), "p2": q(p[1]), "l2": q(l_[1]), "L": q(l_.mean())}
    bwd = {"L": 1.0, "l1": 0.5, "l2": 0.5, "p1": q(ap[0]), "p2": q(ap[1]), "z1": q(az[0]), "z2": q(az[1]),
           "w1": q(aw[0]), "w2": q(aw[1]), "b": q(az.sum())}
    save("w3_hand", {"forward": fwd, "backward": bwd})
    zd = X[:, 0]
    pd = p * (1 - p) * zd
    ld = (-Y / p + (1 - Y) / (1 - p)) * pd
    tang = {"z1": q(zd[0]), "p1": q(pd[0]), "l1": q(ld[0]), "z2": q(zd[1]), "p2": q(pd[1]), "l2": q(ld[1]), "L": q(ld.mean())}
    save("w3_forward_mode", {"tangents": tang, "passes": {"forward": 3, "backward": 1}})


def week4(C):
    # gradient of the Lab 3.1 loss, computed in closed form ((p - y) / 2 per learner)
    T = C["two_learners"]
    X, Y, W, B = np.array(T["x"]), np.array(T["y"]), np.array(T["w"]), T["b"]
    az = (U.sigmoid(X @ W + B) - Y) / 2
    save("w4_torch_grad", [float(az @ X[:, 0]), float(az @ X[:, 1]), float(az.sum())])
    save("w4_fd_worst", 1e-9)  # the order of the step error of central differences at h = 1e-6
    save("w4_overwrite_fails", ["add_mul", "constants_both_sides", "same_node_twice", "random_expressions",
                                "two_learner_loss", "zero_grad"])


def week5(C):
    """Loss histories of Lab 5.1 on the public airfoil data (downloaded; only derived losses are saved)."""
    X, y, source = U.public_data("airfoil")
    if "stand-in" in source:
        raise SystemExit("week 5 references need the real download")
    perm = np.random.default_rng(5).permutation(len(y))
    TR = perm[:1200]
    mu, sd = X[TR].mean(axis=0), X[TR].std(axis=0)
    A = np.column_stack([(X - mu) / sd, np.ones(len(y))])[TR]
    b = y[TR]
    L = lambda A_, w: 0.5 * float(np.mean((A_ @ w - b) ** 2))  # noqa: E731
    G = lambda A_, w, i=None: (A_[i].T @ (A_[i] @ w - b[i]) / len(i)) if i is not None else A_.T @ (A_ @ w - b) / len(b)  # noqa: E731
    ref = L(A, np.linalg.lstsq(A, b, rcond=None)[0])

    def run(step_fn, A_=A):
        w, state, hist = np.zeros(A_.shape[1]), {}, [L(A_, np.zeros(A_.shape[1]))]
        with np.errstate(over="ignore", invalid="ignore"):
            for e in range(200):
                w = step_fn(A_, w, state, e)
                hist.append(L(A_, w))
                if not np.isfinite(hist[-1]):
                    break
        return [h if np.isfinite(h) else float("inf") for h in hist]

    def sgd_step(lr, schedule=None):
        def f(A_, w, st, e):
            order = np.random.default_rng(e).permutation(len(b))
            for i in range(0, len(order), 32):
                rate = lr if schedule is None else lr * schedule(st.setdefault("t", 0))
                w = w - rate * G(A_, w, order[i:i + 32])
                st["t"] = st.get("t", 0) + 1
            return w
        return f

    def mom_step(lr):
        def f(A_, w, st, e):
            st["v"] = 0.9 * st.get("v", 0) + G(A_, w)
            return w - lr * st["v"]
        return f

    batch = {f"lr={lr}": run(lambda A_, w, st, e, lr=lr: w - lr * G(A_, w)) for lr in [0.01, 0.03, 0.1, 0.3, 0.9, 1.0]}
    sgd = {f"lr={lr}": run(sgd_step(lr)) for lr in [0.003, 0.01, 0.03, 0.1, 0.3]}
    mom = {f"lr={lr}": run(mom_step(lr)) for lr in [0.001, 0.01, 0.03, 0.1, 0.3]}
    A_raw = np.column_stack([X[TR], np.ones(len(TR))])
    e = np.linalg.eigvalsh(A_raw.T @ A_raw / len(TR))
    raw = {f"lr={lr}": run(lambda A_, w, st, e_, lr=lr: w - lr * G(A_, w), A_raw) for lr in ["1e-7", "1e-6"] for lr in [float(lr)]}
    raw = {"lr=1e-7": raw.get("lr=1e-07"), "lr=1e-6": raw.get("lr=1e-06")}
    sched = lambda t: 0.1 + 0.9 * t / 100 if t < 100 else 0.5 ** ((t - 100) // 1500)  # noqa: E731
    save("w5_batch", batch)
    save("w5_sgd", sgd)
    save("w5_momentum", mom)
    save("w5_unscaled", {"runs": raw, "lam_max": e[-1], "lam_min": e[0]})
    save("w5_schedule", run(sgd_step(0.3, sched)))

    def diag(h):
        h = np.array(h, dtype=float)
        if not np.isfinite(h[-1]) or h[-1] > h[0]:
            return "diverged"
        if h[-1] > 1.1 * ref:
            return "slow"
        return "noisy" if h[-20:].max() - h[-20:].min() > 0.05 * ref else "converged"
    diags = {}
    for name, runs in (("batch", batch), ("sgd", sgd), ("momentum", mom), ("batch, unscaled", raw)):
        for k, h in runs.items():
            diags[f"{name} {k}"] = diag(h)
    save("w5_diagnoses", diags)


def week6(C):
    S = {
        "bowl": (lambda w: 0.5 * (w[0] ** 2 + 2 * w[1] ** 2), (2.0, 2.0), 0.4, 60, (-3, 3, -3, 3)),
        "narrow": (lambda w: 0.5 * (w[0] ** 2 + 25 * w[1] ** 2), (3.0, 1.0), 0.085, 60, (-4, 4, -1.5, 1.5)),
        "valley": (lambda w: (1 - w[0]) ** 2 + 5 * (w[1] - w[0] ** 2) ** 2, (-1.0, 2.0), 0.02, 1500, (-1.5, 1.8, -0.5, 2.5)),
        "saddle": (lambda w: w[0] ** 2 - w[1] ** 2 + 0.25 * w[1] ** 4, (1.0, 0.0), 0.1, 200, (-2, 2, -2.2, 2.2)),
        "bumpy": (lambda w: 0.1 * w[0] ** 2 + 0.5 * np.sin(2 * w[0]) + w[1] ** 2, (3.0, 1.0), 0.1, 300, (-4, 4, -2, 2)),
        "plateau": (lambda w: 1 - np.exp(-(w[0] ** 2 + w[1] ** 2) / 8), (8.0, 8.0), 0.5, 300, (-10, 10, -10, 10)),
    }

    def hess(f, w, h=1e-4):
        w = np.asarray(w, dtype=float)
        E = np.eye(2) * h
        H = np.array([[(f(w + E[i] + E[j]) - f(w + E[i] - E[j]) - f(w - E[i] + E[j]) + f(w - E[i] - E[j])) / (4 * h * h)
                       for j in range(2)] for i in range(2)])
        return (H + H.T) / 2

    def kind(H):
        e = np.linalg.eigvalsh(H)
        if np.any(np.abs(e) < 1e-6):
            return "flat"
        return "minimum" if np.all(e > 0) else "maximum" if np.all(e < 0) else "saddle"

    def trace(f, s, lr, n):
        w, path = np.array(s, dtype=float), [np.array(s, dtype=float)]
        with np.errstate(over="ignore", invalid="ignore"):
            for _ in range(n):
                w = w - lr * U.central_diff(f, w)
                path.append(w.copy())
                if not np.all(np.isfinite(w)):
                    break
        return path

    pts = {"bowl": (0, 0), "narrow": (0, 0), "valley": (1, 1), "saddle": (0, 0), "saddle, side": (0, 2 ** 0.5)}
    save("w6_points", {k: [np.round(np.linalg.eigvalsh(hess(S[k.split(",")[0]][0], p)), 3).tolist(),
                           kind(hess(S[k.split(",")[0]][0], p))] for k, p in pts.items()})
    convex = {}
    for k, (f, s, lr, n, box) in S.items():
        m = min(np.linalg.eigvalsh(hess(f, (x, y)))[0] for x in np.linspace(box[0], box[1], 15) for y in np.linspace(box[2], box[3], 15))
        convex[k] = bool(m > -1e-6)
    save("w6_convex", convex)
    fn = S["narrow"][0]
    save("w6_edge", {str(lr): float(fn(trace(fn, (3.0, 1.0), lr, 60)[-1])) for lr in (0.05, 0.079, 0.081)})
    fs = S["saddle"][0]
    save("w6_ridge", [trace(fs, (1.0, 0.0), 0.1, 200)[-1].tolist(), trace(fs, (1.0, 0.001), 0.1, 200)[-1].tolist()])
    notes, traces = {}, {}
    for k, (f, s, lr, n, box) in S.items():
        path = trace(f, s, lr, n)
        step = max(1, len(path) // 150)
        traces[k] = [[float(v) for v in p] for p in path[::step] if np.all(np.isfinite(p)) and np.abs(p).max() < 1e3]
        end = path[-1]
        if not np.all(np.isfinite(end)) or np.abs(end).max() > 1e3:
            o, g = "diverged", None
        elif np.linalg.norm(end - np.array(s)) < 0.01:
            o, g = "plateau", None
        elif kind(hess(f, end)) == "saddle":
            o, g = "saddle", None
        else:
            xs, ys = np.meshgrid(np.linspace(box[0], box[1], 400), np.linspace(box[2], box[3], 400))
            o, g = "minimum", bool(f(end) - f(np.array([xs, ys])).min() < 1e-3)
        notes[k] = {"convex": convex[k], "outcome": o, "global": g}
    save("w6_annotations", notes)
    save("w6_traces", traces)


def week7(C):
    def lse(Z):
        m = Z.max(axis=1, keepdims=True)
        return (m + np.log(np.exp(Z - m).sum(axis=1, keepdims=True)))[:, 0]

    sm = lambda Z: np.exp(Z - lse(Z)[:, None])  # noqa: E731
    ce = lambda Z, y: float(np.mean(lse(Z) - Z[np.arange(len(y)), y]))  # noqa: E731
    Zb, yb = np.array(C["overflow_logits"]), np.array([1, 2, 0])
    save("w7_stable", {"softmax": np.round(sm(Zb), 6).tolist(), "cross_entropy": ce(Zb, yb)})

    rng = np.random.default_rng(7)
    Zs, ys = rng.normal(0, 2, (5, 3)), np.array([0, 2, 1, 1, 0])
    Xs, Ws = rng.normal(0, 1, (5, 4)), rng.normal(0, 0.5, (4, 3))
    Y = np.eye(3)[ys]
    gZ = (sm(Zs) - Y) / 5
    nZ = U.central_diff(lambda z: ce(z.reshape(5, 3), ys), Zs.ravel()).reshape(5, 3)
    gW = Xs.T @ (sm(Xs @ Ws) - Y) / 5
    nW = U.central_diff(lambda w: ce(Xs @ w.reshape(4, 3), ys), Ws.ravel()).reshape(4, 3)
    save("w7_gradcheck", [U.rel_err(gZ, nZ), U.rel_err(gW, nW)])

    L = U.learners()
    Xr = np.column_stack([L["entry_score"], L["attendance"], L["weekly_hours"]]).astype(float)
    y = L["level"]
    perm = np.random.default_rng(102).permutation(len(y))
    TR, TE = perm[:450], perm[450:]
    X = np.column_stack([(Xr - Xr[TR].mean(axis=0)) / Xr[TR].std(axis=0), np.ones(len(y))])
    Xtr, ytr, Xte, yte = X[TR], y[TR], X[TE], y[TE]

    def train(lam, steps=2000, lr=0.5):
        W, Yt, hist = np.zeros((4, 3)), np.eye(3)[ytr], []
        for t in range(1, steps + 1):
            W = W - lr * (Xtr.T @ (sm(Xtr @ W) - Yt) / len(ytr) + lam * W)
            if t % 50 == 0:
                hist.append((t, ce(Xtr @ W, ytr), float(np.linalg.norm(W))))
        return W, hist

    W, hist = train(0.0)
    save("w7_plain", {"train": ce(Xtr @ W, ytr), "held_out": ce(Xte @ W, yte),
                      "accuracy": float(np.mean((Xte @ W).argmax(axis=1) == yte)), "history": hist})
    rows = []
    for lam in [0.0, 1e-3, 1e-2, 1e-1, 1.0]:
        W, h = train(lam)
        rows.append([lam, float(np.linalg.norm(W)), ce(Xtr @ W, ytr), ce(Xte @ W, yte),
                     float(np.mean((Xte @ W).argmax(axis=1) == yte))])
        if lam == 1e-2:
            save("w7_penalised_history", h)
    save("w7_sweep", rows)
    best = min(rows, key=lambda r: r[3])
    save("w7_compare", [best[0], bool(any(r[3] < rows[0][3] for r in rows[1:]))])


def main() -> None:
    C = U.case()
    weeks = sys.argv[1:] or ["1"]
    for w in weeks:
        globals()[f"week{w}"](C)


if __name__ == "__main__":
    main()
