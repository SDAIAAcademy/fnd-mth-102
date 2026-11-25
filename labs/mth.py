"""Shared helpers for the labs of Calculus and Optimisation for Machine Learning.

    import mth as U

Quick mode: set MTH_QUICK=1 and U.pick(full, quick) returns the smaller value (fewer steps,
fewer seeds). Safety nets: U.reference(name) returns the reference answer of an exercise, so a
notebook runs to its end when an exercise is not done yet. Outputs (figures, logs) go to
labs/out/, which git ignores.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

LABS = Path(__file__).resolve().parent
DATA = LABS / "data"
OUT = LABS / "out"
CACHE = DATA / "cache"
QUICK = os.environ.get("MTH_QUICK", "") not in ("", "0")


def pick(full, quick):
    """The full value, or the quick one when MTH_QUICK=1."""
    return quick if QUICK else full


def reference(name: str):
    """The reference answer of an exercise, for the safety-net cells."""
    value = json.loads((DATA / "reference" / f"{name}.json").read_text(encoding="utf-8"))
    print(f"(safety net: using the reference value of {name})")
    return value


def out(name: str) -> Path:
    """A path in labs/out/ for a figure or a log."""
    OUT.mkdir(exist_ok=True)
    return OUT / name


def case() -> dict:
    """The case file of the adult learning charity (synthetic)."""
    return json.loads((DATA / "case.json").read_text(encoding="utf-8"))


def learners() -> dict:
    """The 600 synthetic learners as NumPy arrays, one per column."""
    with open(DATA / "learners.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    cols = {k: [r[k] for r in rows] for k in rows[0]}
    res = {"code": np.array(cols["code"]), "district": np.array(cols["district"])}
    for k in ("attendance", "weekly_hours"):
        res[k] = np.array(cols[k], dtype=float)
    for k in ("entry_score", "completed", "level"):
        res[k] = np.array(cols[k], dtype=int)
    return res


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def central_diff(f, x, h=1e-6):
    """Central-difference gradient of a scalar function f at x (a float or an array)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        return (f(x + h) - f(x - h)) / (2 * h)
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e.flat[i] = h
        g.flat[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def rel_err(a, b) -> float:
    """Relative error between two gradients: |a - b| / max(|a|, |b|, 1e-12)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-12))


def plt():
    """matplotlib.pyplot with the course defaults; figures are saved, never needed on screen."""
    import matplotlib

    import sys

    if "ipykernel" not in sys.modules and not os.environ.get("MPLBACKEND"):
        matplotlib.use("Agg")  # run as a script: save figures, open no window
    import matplotlib.pyplot as p

    p.rcParams.update({"figure.figsize": (6.4, 4.0), "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})
    return p


# ---------------------------------------------------------------- public data (UCI, CC BY 4.0)

PUBLIC = {
    "airfoil": {
        "url": "https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip",
        "member": "airfoil_self_noise.dat",
        "sha256": "74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3",
        "cite": "Airfoil Self-Noise, UCI Machine Learning Repository, doi 10.24432/C5VW2C, CC BY 4.0",
        "columns": ["frequency", "angle", "chord", "velocity", "thickness", "sound_level"],
    },
    "banknote": {
        "url": "https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip",
        "member": "data_banknote_authentication.txt",
        "sha256": "d0539aaed2139ba7a587b3e34fb345ce503ff7d5d33dbf9912d8e195ce425cb9",
        "cite": "Banknote Authentication, UCI Machine Learning Repository, doi 10.24432/C55P57, CC BY 4.0",
        "columns": ["variance", "skewness", "curtosis", "entropy", "label"],
    },
}


def _download(name: str) -> bytes:
    spec = PUBLIC[name]
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / spec["member"]
    if cached.exists():
        return cached.read_bytes()
    last = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(spec["url"], timeout=30) as r:
                blob = r.read()
            raw = zipfile.ZipFile(io.BytesIO(blob)).read(spec["member"])
            break
        except Exception as e:  # network or archive error: try again
            last = e
    else:
        raise ConnectionError(f"could not download {name}: {last}")
    digest = hashlib.sha256(raw).hexdigest()
    if spec["sha256"] and digest != spec["sha256"]:
        raise ValueError(f"{name}: the file changed upstream (sha256 {digest[:12]}...); tell the trainer")
    cached.write_bytes(raw)
    return raw


def _stand_in(name: str) -> np.ndarray:
    """A seeded synthetic stand-in with the same columns, used only when the download fails."""
    rng = np.random.default_rng(7)
    if name == "airfoil":
        n = 1503
        X = np.column_stack([rng.choice([800, 1600, 3150, 6300], n), rng.uniform(0, 22, n),
                             rng.uniform(0.025, 0.3, n), rng.choice([31.7, 39.6, 55.5, 71.3], n),
                             rng.uniform(0.0004, 0.06, n)])
        y = 132 - 0.0012 * X[:, 0] - 0.4 * X[:, 1] - 35 * X[:, 2] + 0.08 * X[:, 3] - 150 * X[:, 4] + rng.normal(0, 4.5, n)
        return np.column_stack([X, y])
    n = 1372
    y = rng.integers(0, 2, n)
    X = rng.normal(0, 1, (n, 4)) * [2.8, 5.8, 4.3, 2.1] + np.outer(1 - 2 * y, [2.2, 3.0, -1.0, 0.3])
    return np.column_stack([X, y])


def public_data(name: str) -> tuple[np.ndarray, np.ndarray, str]:
    """(X, y, source) for a public UCI dataset; downloaded once into labs/data/cache/ (git ignores it).

    If the download fails, a seeded synthetic stand-in is returned and the source says so loudly:
    numbers from a stand-in run are not the public-data results.
    """
    try:
        raw = _download(name).decode("utf-8")
        sep = None if name == "airfoil" else ","
        arr = np.array([[float(v) for v in line.split(sep)] for line in raw.splitlines() if line.strip()])
        source = PUBLIC[name]["cite"]
    except Exception as e:
        print(f"!! {name}: download failed ({e}). Using a SYNTHETIC STAND-IN: results are not the public data's.")
        arr = _stand_in(name)
        source = f"synthetic stand-in for {name} (download failed)"
    return arr[:, :-1], arr[:, -1], source
