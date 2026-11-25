"""Write the reference answers of the labs' safety nets into labs/data/reference/.

    cd labs
    uv run python _tools/make_reference.py

Computed from the case data and the mathematics, without running the labs. Floats are kept to
eight significant digits, so no file holds a long run of digits.
"""
from __future__ import annotations

import json
import math
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
    (REF / f"{name}.json").write_text(json.dumps(r(value), indent=1) + "\n", encoding="utf-8", newline="\n")
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


def main() -> None:
    C = U.case()
    weeks = sys.argv[1:] or ["1"]
    for w in weeks:
        globals()[f"week{w}"](C)


if __name__ == "__main__":
    main()
