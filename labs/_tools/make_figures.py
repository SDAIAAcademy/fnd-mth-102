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


if __name__ == "__main__":
    for w in sys.argv[1:] or ["1"]:
        globals()[f"week{w}"]()
