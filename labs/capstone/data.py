"""The capstone's data: Banknote Authentication (UCI, CC BY 4.0), split and standardised.

    import sys; sys.path.insert(0, "labs/capstone"); from data import load
    Xtr, ytr, Xte, yte, source = load()

Four numeric features of images of banknotes, and a label 0 or 1. The split is fixed by a seed
(70% training, 30% held out); the features are standardised with the training rows' mean and
standard deviation. The first call downloads the file into labs/data/cache/ (git ignores it) and
checks its SHA-256. If the download fails, a seeded synthetic stand-in is used and `source` says
so: a report built on the stand-in must say so too.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mth as U  # noqa: E402


def load(seed: int = 102, train_share: float = 0.7):
    """(X_train, y_train, X_held_out, y_held_out, source); arrays of floats, labels 0.0 or 1.0."""
    X, y, source = U.public_data("banknote")
    perm = np.random.default_rng(seed).permutation(len(y))
    n = int(train_share * len(y))
    tr, te = perm[:n], perm[n:]
    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0)
    X = (X - mu) / sd
    return X[tr], y[tr], X[te], y[te], source


if __name__ == "__main__":
    Xtr, ytr, Xte, yte, source = load()
    print(source)
    print(f"training {Xtr.shape}, held out {Xte.shape}, share of label 1: {ytr.mean():.3f} and {yte.mean():.3f}")
