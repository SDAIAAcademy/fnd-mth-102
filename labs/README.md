# Calculus and Optimisation for Machine Learning: labs

## Quick start

- Google Colab: open a notebook from the course page; its first cell clones this repository.
  Colab already has every library the labs use.
- Locally, with [uv](https://docs.astral.sh/uv/) installed, from this folder:

```bash
uv sync                               # Python 3.12, NumPy, SymPy, matplotlib, PyTorch (CPU), JupyterLab
uv run python _tools/py2nb.py         # writes the notebooks from the sources in _src/
uv run jupyter lab 00_setup_check.ipynb
```

Run `00_setup_check.ipynb` once first. No lab calls an AI model, so no key is needed. Set
`MTH_QUICK=1` for a short run (fewer steps and seeds).

| Unit | Session | Notebook | AI level |
|---|---|---|---|
| Week 1 | Derivatives as local lines | [`W1_1_derivatives.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/fnd-mth-102/blob/main/labs/W1_1_derivatives.ipynb) | tutor |
| Week 2 | Gradients and Jacobians | [`W2_1_gradients.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/fnd-mth-102/blob/main/labs/W2_1_gradients.ipynb) | tutor |
| Week 3 | Graphs and the chain rule | [`W3_1_backprop_by_hand.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/fnd-mth-102/blob/main/labs/W3_1_backprop_by_hand.ipynb) | off |
| Week 4 | A reverse-mode engine | [`W4_1_autodiff_engine.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/fnd-mth-102/blob/main/labs/W4_1_autodiff_engine.ipynb) | tutor |
| Week 5 | Descent and loss curves | [`W5_1_lr_sweeps.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/fnd-mth-102/blob/main/labs/W5_1_lr_sweeps.ipynb) | tutor |
| Week 6 | Convexity and curvature | [`W6_1_loss_surfaces.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/fnd-mth-102/blob/main/labs/W6_1_loss_surfaces.ipynb) | tutor |
| Week 7 | Stable losses and penalties | [`W7_1_stable_softmax.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/fnd-mth-102/blob/main/labs/W7_1_stable_softmax.ipynb) | tutor |
| Week 8 | Constraints and prices | [`W8_1_budget_allocation.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/fnd-mth-102/blob/main/labs/W8_1_budget_allocation.ipynb) | tutor |

The tutor's instruction text is in [`TUTOR.md`](TUTOR.md).

## What else is here

| Path | What it is |
|---|---|
| `mth.py` | The labs' shared helpers: data loading, quick mode, safety nets, the public-data download |
| `data/` | The charity's synthetic data (`learners.csv`, `case.json`) and the safety nets' reference answers |
| `prework/README.md` | The school calculus refresher (80 minutes, before week 1) |
| `problem_sets/` | Problem sets PS1 to PS8 (50 minutes each, graded) and `check.py`, which checks them by property |
| `engine_tests.py` | The test suite of the week 4 engine, reused by the capstone |
| `capstone/` | The capstone brief and its data loader |
| `solutions/` | The reference notebooks and the labs' answer key |
| `_src/`, `_tools/` | Notebook sources, the data generator and the figure script (not published on the site) |

## Public data

Lab 5.1 and the capstone download two public datasets from the UCI Machine Learning Repository,
both under the Creative Commons Attribution 4.0 licence, into `data/cache/`, which git ignores:
[Airfoil Self-Noise](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise) (doi
10.24432/C5VW2C) and [Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication)
(doi 10.24432/C55P57). `mth.py` checks each file's SHA-256. If the download fails, it uses a
seeded synthetic stand-in and says so in capitals: results from a stand-in are not the public
data's results.

The notebooks are generated from the percent-format sources in `labs/_src/` by `labs/_tools/py2nb.py` (`uv run python _tools/py2nb.py` from `labs/`), which also writes the solutions to `labs/solutions/`. Edit the sources, not the notebooks.
