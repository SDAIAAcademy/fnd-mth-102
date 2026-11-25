# %% [markdown]
# # Setup check
#
# Run this once before Lab 1.1. It imports every library the labs use, reads the synthetic case
# data and downloads the two public datasets into `data/cache/` (git ignores that folder). It
# prints what is missing. No lab of this course calls an AI model, so no key is needed.

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
        os.chdir("..")
sys.path.insert(0, os.getcwd())

# %%
import importlib

missing = []
for name in ("numpy", "sympy", "matplotlib", "torch"):
    try:
        mod = importlib.import_module(name)
        print(f"{name:11} {getattr(mod, '__version__', '?')}")
    except ImportError:
        missing.append(name)
print("missing:", ", ".join(missing) if missing else "nothing")

# %%
import mth as U

L = U.learners()
print("learners:", len(L["code"]), "| districts:", sorted(set(L["district"])))
for name in ("airfoil", "banknote"):
    X, y, source = U.public_data(name)
    print(f"{name}: {X.shape[0]} rows, {X.shape[1]} features | {source}")
