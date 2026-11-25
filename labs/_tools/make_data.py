"""Write the synthetic data of the adult learning charity into labs/data/.

    cd labs
    uv run python _tools/make_data.py

Fixed seed (102): the same files on every run. Every number is rounded so that no file holds a
long run of digits. Learners carry codes only (LR-0001 ...): no names, no identity numbers, no
phone numbers. The reference answers of the labs' safety nets are written by
_tools/make_reference.py, which reads these files.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

LABS = Path(__file__).resolve().parents[1]
DATA = LABS / "data"
SEED = 102
N = 600
DISTRICTS = ["D1", "D2", "D3", "D4", "D5"]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def learners(rng: np.random.Generator) -> list[dict]:
    # districts D4 and D5 hold the learners furthest behind: lower entry scores
    district = rng.choice(5, size=N, p=[0.24, 0.24, 0.22, 0.16, 0.14])
    shift = np.array([8.0, 5.0, 2.0, -12.0, -16.0])[district]
    entry = np.clip(np.round(rng.normal(52 + shift, 17)), 0, 100)
    attendance = np.clip(np.round(rng.beta(5, 2, size=N) - 0.02 * (district >= 3), 3), 0.05, 1.0)
    hours = np.clip(np.round(rng.normal(4.0, 1.3, size=N), 1), 1.0, 8.0)
    # completion: a logistic model of attendance, entry score and weekly hours, plus noise
    z = -4.2 + 4.8 * attendance + 0.025 * entry + 0.18 * hours
    completed = (rng.random(N) < sigmoid(z)).astype(int)
    # placement level: thresholds on the entry score, with no noise (planted: separable)
    level = np.where(entry < 40, 0, np.where(entry < 70, 1, 2))
    rows = []
    for i in range(N):
        rows.append({
            "code": f"LR-{i + 1:04d}",
            "district": DISTRICTS[district[i]],
            "attendance": f"{attendance[i]:.3f}",
            "entry_score": int(entry[i]),
            "weekly_hours": f"{hours[i]:.1f}",
            "completed": int(completed[i]),
            "level": int(level[i]),
        })
    return rows


def main() -> None:
    rng = np.random.default_rng(SEED)
    DATA.mkdir(exist_ok=True)
    rows = learners(rng)
    with open(DATA / "learners.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    case = {
        "about": "Synthetic. The adult learning charity: five districts D1 to D5 of one city. Budget figures are in thousands.",
        # Week 1 and 2: the completion model the data analyst fitted last term, on attendance a (0 to 1)
        # and weekly hours h; p(a, h) = sigmoid(w_a * a + w_h * h + b)
        "completion_model": {"w_a": 4.0, "w_h": 0.3, "b": -3.5},
        # Week 1, 2 and 8: enrolments E_i(x) = k_i * log(1 + x / s_i) for outreach budget x (thousands)
        "districts": {
            "D1": {"k": 60.0, "s": 10.0, "behind_share": 0.18},
            "D2": {"k": 52.0, "s": 12.0, "behind_share": 0.22},
            "D3": {"k": 45.0, "s": 8.0, "behind_share": 0.27},
            "D4": {"k": 18.0, "s": 15.0, "behind_share": 0.46},
            "D5": {"k": 14.0, "s": 20.0, "behind_share": 0.55},
        },
        "budget": 120.0,
        "floor": 10.0,
        # Week 3: two learners for the hand backward pass (features already scaled)
        "two_learners": {
            "codes": ["LR-0007", "LR-0042"],
            "x": [[0.9, 0.6], [0.4, 0.2]],
            "y": [1, 0],
            "w": [1.5, -2.0],
            "b": 0.5,
        },
        # Week 7: logits an over-confident placement model gave three learners
        "overflow_logits": [[812.0, 790.0, 640.0], [15.0, 820.0, 805.0], [-760.0, -731.0, -745.0]],
    }
    (DATA / "case.json").write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("wrote", (DATA / "learners.csv").relative_to(LABS), len(rows), "learners")
    print("wrote", (DATA / "case.json").relative_to(LABS))


if __name__ == "__main__":
    main()
