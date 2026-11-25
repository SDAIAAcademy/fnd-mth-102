# Capstone brief: a two-layer network on your own engine

Train a small neural network on public data with the reverse-mode engine you wrote in week 4,
check its gradients, compare three optimisers across seeds, and report every run, including the
ones that failed. Then defend it in a ten-minute oral check.

**AI level: AI-full.** You may use any assistant for any part of the work. The oral check is where
you show that you understand what you submit: you will be asked to trace a backward pass through
your own engine and to explain a failed run from your own log.

## The data

Banknote Authentication (UCI Machine Learning Repository, doi 10.24432/C55P57, CC BY 4.0): four
numeric features of images of banknotes and a label 0 or 1. Load it with the supplied loader, which
fixes the split (70% training, 30% held out) and standardises the features:

```python
import sys; sys.path.insert(0, "labs/capstone")
from data import load
Xtr, ytr, Xte, yte, source = load()
```

If `source` says the download failed and a synthetic stand-in was used, try again later; a report
built on the stand-in must say so on its first line.

## What to build

1. **The engine.** Your own `Value` class from Lab 4.1 (scalar reverse mode), extended as you need.
   It must pass the course test suite: `engine_tests.run_tests(Value)`.
2. **The network.** Four inputs, one hidden layer of your choice of width and activation, one output
   logit. The loss is the mean binary cross-entropy computed from the logit in the stable form of
   week 7, never as the log of a probability.
3. **A gradient check on the network.** Compare your engine's gradients with central differences
   for at least five parameters on a small batch; report the worst relative error.
4. **Training.** Mini-batch training with three optimisers, for example plain SGD, momentum and an
   adaptive method, each across at least three seeds. Choose each optimiser's learning rate with a
   short sweep and report the sweep too.
5. **The log.** One row per run: optimiser, learning rate, seed, epochs, final training loss,
   held-out loss, held-out accuracy, and whether the run failed. Keep the loss curve of every run.
6. **The report** (two pages at most): what you built, the gradient check, the comparison with
   mean and spread across seeds, and a written diagnosis of each failed run from its loss curve and
   log, using the words of week 5 (diverged, slow, noisy) and the reason you give for it.

A run that diverged or stalled is evidence, not an embarrassment: you will have some, and the
rubric gives marks for reporting and diagnosing them.

## Submission

A link to a public GitHub repository with the engine, its tests and gradient checks, the training
code, the log of every run with its seed, and the report. Anyone must be able to reproduce your
numbers from the seeds.

## Rubric (capstone engine and report, 30% of the course grade)

| Criterion | Weight |
|---|---|
| Engine correct: passes the course test suite and the gradient checks on the network | 25 |
| Network trained on the public data with a held-out split, reproducible from the seeds | 20 |
| Three optimisers compared across at least three seeds, with loss curves and held-out accuracy | 25 |
| Every run, seed and failure reported, not the best result alone | 15 |
| A written diagnosis of each failed run from its loss curve and log | 15 |

## The oral check (10% of the course grade)

Ten minutes, questions included. You will be asked to:

- trace a backward pass through your engine for a small expression the trainer writes down, and
  show where each gradient is accumulated;
- open your log, pick one failed run, and explain from its curve what went wrong and what you
  changed;
- answer one question about a choice in your code.

## The day

| Block | Minutes |
|---|---|
| Brief and planning clinic | 30 |
| Build, train and write | 255 |
| Gradient check and a test run of the oral, in pairs | 30 |
| Oral check (yours) | 10 |
| Changeover | 5 |
| Written feedback from the trainer | 30 |

Plan for the network to train in minutes, not hours: a scalar engine is slow, and a small hidden
layer with a few tens of epochs is enough.
