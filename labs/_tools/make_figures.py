"""Draw the figures of the slides into assets/ as SVG, computed here, never typed in.

    cd labs
    uv run python _tools/make_figures.py [week ...]

Slides before a lab show parallel examples (another district, another function, other numbers);
the lab's own results reach the slides only in the debrief of the following week.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABS = Path(__file__).resolve().parents[1]
ASSETS = LABS.parent / "assets"
sys.path.insert(0, str(LABS))
import mth as U  # noqa: E402

plt.rcParams.update({
    "svg.fonttype": "none", "svg.hashsalt": "fnd-mth-102", "font.size": 13, "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False, "font.family": "sans-serif",
})
TEAL, CORAL, NAVY, PURPLE, AMBER = "#0f8b8d", "#e4572e", "#1d2b53", "#7b61ff", "#c98a00"


def save(fig, name: str) -> None:
    ASSETS.mkdir(exist_ok=True)
    path = ASSETS / name
    fig.savefig(path, format="svg", metadata={"Date": None, "Creator": None}, bbox_inches="tight")
    plt.close(fig)
    # keep three decimals in coordinates: no long digit runs in a committed file
    import re
    txt = path.read_text(encoding="utf-8")
    txt = re.sub(r"(\d+\.\d{3})\d+", r"\1", txt)
    path.write_text(txt, encoding="utf-8", newline="\n")
    print("wrote", path.relative_to(LABS.parent))


def week1() -> None:
    C = U.case()
    D3 = C["districts"]["D3"]
    E = lambda x: D3["k"] * np.log1p(x / D3["s"])  # noqa: E731
    x0 = 12.0
    d1, d2 = D3["k"] / (D3["s"] + x0), -D3["k"] / (D3["s"] + x0) ** 2
    xs = np.linspace(0, 50, 200)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(xs, E(xs), color=NAVY, lw=2.5, label="enrolments E(x), district D3")
    ax.plot(xs, E(x0) + d1 * (xs - x0), color=CORAL, lw=2, ls="--", label="tangent line at x = 12")
    ax.plot(xs, E(x0) + d1 * (xs - x0) + 0.5 * d2 * (xs - x0) ** 2, color=TEAL, lw=2, ls=":", label="quadratic at x = 12")
    ax.plot([x0], [E(x0)], "o", color=CORAL, ms=8)
    ax.set_xlabel("outreach budget x (thousands)"), ax.set_ylabel("enrolments")
    ax.set_ylim(0, 140), ax.legend(loc="lower right", frameon=False)
    save(fig, "w1-tangent-d3.svg")

    g = lambda x: np.exp(0.5 * x)  # noqa: E731
    x0, exact = 1.0, 0.5 * np.exp(0.5)
    HS = np.logspace(-1, -12, 23)
    fwd = [abs((g(x0 + h) - g(x0)) / h - exact) for h in HS]
    cen = [abs((g(x0 + h) - g(x0 - h)) / (2 * h) - exact) for h in HS]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.loglog(HS, np.maximum(fwd, 1e-17), "o-", color=CORAL, ms=4, label="forward difference")
    ax.loglog(HS, np.maximum(cen, 1e-17), "s-", color=TEAL, ms=4, label="central difference")
    ax.set_xlabel("step h"), ax.set_ylabel("absolute error of g'(1)"), ax.invert_xaxis()
    ax.legend(frameon=False)
    save(fig, "w1-error-v-exp.svg")
    print("exp example: best forward h", HS[int(np.argmin(fwd))], "best central h", HS[int(np.argmin(cen))],
          "min errors", min(fwd), min(cen))


def week2() -> None:
    # parallel example on two inputs scaled to 0..1: r(u, v) = sigmoid(3u + 2v - 2.5), not the lab's model
    r = lambda u, v: U.sigmoid(3 * u + 2 * v - 2.5)  # noqa: E731
    A, H = np.meshgrid(np.linspace(0, 1, 120), np.linspace(0, 1, 120))
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    cs = ax.contour(A, H, r(A, H), levels=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], colors=NAVY, linewidths=1.2)
    ax.clabel(cs, fmt="%.1f", fontsize=10)
    for u0, v0 in [(0.15, 0.2), (0.45, 0.45), (0.75, 0.7), (0.3, 0.85), (0.85, 0.15)]:
        s = r(u0, v0)
        g = np.array([3 * s * (1 - s), 2 * s * (1 - s)])
        ax.annotate("", xy=(u0 + 0.22 * g[0], v0 + 0.22 * g[1]), xytext=(u0, v0),
                    arrowprops=dict(arrowstyle="->", color=CORAL, lw=2))
        ax.plot([u0], [v0], "o", color=CORAL, ms=5)
    ax.set_aspect("equal"), ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.set_xlabel("attendance share u"), ax.set_ylabel("hours, as a share of 8, v")
    save(fig, "w2-gradient-contours.svg")
    s = r(0.45, 0.45)
    print("r(0.45,0.45)", s, "grad", 3 * s * (1 - s), 2 * s * (1 - s))


def week5() -> None:
    # parallel example: the quadratic f(u, v) = 0.5 (u^2 + 10 v^2), curvatures 1 and 10, start (10, 1)
    lam = np.array([1.0, 10.0])
    f = lambda w: 0.5 * float(np.sum(lam * w * w))  # noqa: E731

    def gd(lr, steps=60, beta=0.0):
        w, v, hist, path = np.array([10.0, 1.0]), np.zeros(2), [], []
        for _ in range(steps + 1):
            hist.append(f(w)), path.append(w.copy())
            v = beta * v + lam * w
            w = w - lr * v
        return hist, np.array(path)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    styles = [(0.01, "slow: rate 0.01", NAVY), (0.15, "converges: rate 0.15", TEAL),
              (0.195, "oscillates: rate 0.195", AMBER), (0.205, "diverges: rate 0.205", CORAL)]
    for lr, label, col in styles:
        h, _ = gd(lr)
        ax.semilogy(h, color=col, lw=2, label=label)
        print(label, "loss after 60 steps", h[-1])
    ax.set_xlabel("step"), ax.set_ylabel("loss (log scale)"), ax.set_ylim(1e-6, 1e4)
    ax.legend(frameon=False, fontsize=11)
    save(fig, "w5-curve-shapes.svg")

    U_, V_ = np.meshgrid(np.linspace(-11, 11, 200), np.linspace(-1.6, 1.6, 200))
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.contour(U_, V_, 0.5 * (U_ ** 2 + 10 * V_ ** 2), levels=12, colors="#9aa5b1", linewidths=0.8)
    _, p1 = gd(0.04, 25)
    _, p2 = gd(0.04, 25, beta=0.8)
    ax.plot(p1[:, 0], p1[:, 1], "o-", color=CORAL, ms=3, lw=1.5, label="gradient descent, rate 0.04")
    ax.plot(p2[:, 0], p2[:, 1], "s-", color=TEAL, ms=3, lw=1.5, label="momentum 0.8, rate 0.04")
    ax.set_xlabel("u (flat direction)"), ax.set_ylabel("v (steep)"), ax.legend(frameon=False, fontsize=10, loc="lower left")
    save(fig, "w5-momentum-path.svg")
    print("after 25 steps: gd", f(p1[-1]), "momentum", f(p2[-1]))


def week6() -> None:
    # parallel example: f = 0.5 x^2 - 0.5 y^2 + 0.1 y^4, saddle at 0, minima at y = +-sqrt(2.5)
    f = lambda x, y: 0.5 * x ** 2 - 0.5 * y ** 2 + 0.1 * y ** 4  # noqa: E731
    g = lambda w: np.array([w[0], -w[1] + 0.4 * w[1] ** 3])  # noqa: E731
    X_, Y_ = np.meshgrid(np.linspace(-2.5, 2.5, 200), np.linspace(-2.2, 2.2, 200))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.contour(X_, Y_, f(X_, Y_), levels=18, colors="#9aa5b1", linewidths=0.8)
    for start, col, label in [((2.0, 0.0), CORAL, "start (2, 0): stops at the saddle"),
                              ((2.0, 0.01), TEAL, "start (2, 0.01): escapes to a minimum")]:
        w, path = np.array(start), [np.array(start)]
        for _ in range(120):
            w = w - 0.2 * g(w)
            path.append(w.copy())
        p = np.array(path)
        ax.plot(p[:, 0], p[:, 1], ".-", color=col, ms=3, lw=1.5, label=label)
        print(label, "ends at", np.round(p[-1], 4))
    ax.plot([0], [0], "x", color=NAVY, ms=10, mew=2)
    ax.set_aspect("equal"), ax.set_xlabel("x"), ax.set_ylabel("y"), ax.legend(frameon=False, fontsize=10, loc="lower right")
    save(fig, "w6-saddle-escape.svg")


def week7() -> None:
    # parallel example: one weight c on four separable points x = -2, -1, 1, 2 with labels 0, 0, 1, 1
    x, s = np.array([-2.0, -1.0, 1.0, 2.0]), np.array([-1.0, -1.0, 1.0, 1.0])
    cs = np.linspace(0, 8, 400)
    plain = np.array([np.mean(np.logaddexp(0.0, -s * c * x)) for c in cs])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(cs, plain, color=CORAL, lw=2.5, label="plain loss: falls for ever")
    for lam, col in [(0.02, PURPLE), (0.1, TEAL)]:
        pen = plain + 0.5 * lam * cs ** 2
        i = int(np.argmin(pen))
        ax.plot(cs, pen, color=col, lw=2, label=f"with penalty, lam = {lam}: minimum at c = {cs[i]:.2f}")
        ax.plot([cs[i]], [pen[i]], "o", color=col, ms=7)
        print(f"lam {lam}: minimum at c = {cs[i]:.3f}, loss {pen[i]:.4f}")
    print("plain loss at c = 8:", plain[-1])
    ax.set_xlabel("weight c"), ax.set_ylabel("loss"), ax.set_ylim(0, 0.8), ax.legend(frameon=False, fontsize=10)
    save(fig, "w7-penalty-shape.svg")


if __name__ == "__main__":
    for w in sys.argv[1:] or ["1"]:
        globals()[f"week{w}"]()
