<!-- academy:title -->
# Calculus and Optimisation for Machine Learning

التفاضل والتكامل والأمثلة الرياضية لتعلم الآلة

Data & AI Foundations Track · SDAIA Academy · FND-MTH-102 · Level 100

Differentiate by hand and in code, build a small reverse-mode engine, and train models with gradient descent you can diagnose

Participants work through the calculus that trains models: derivatives as local linear approximations, gradients, Jacobians and the chain rule on computational graphs. They backpropagate by hand, then write a small reverse-mode automatic differentiation engine and check it against a framework. They run batch, stochastic and momentum gradient descent, read loss curves, and handle convexity and constraints. The capstone trains a small network with their own engine; tensor-level differentiation and modern optimisers follow in SPC-DLR-302.

**Format:** Extended: 8 weeks part-time, one session and one lab a week, weekly graded problem sets and a capstone

**Course page:** https://sdaiaacademy.github.io/fnd-mth-102/
<!-- /academy:title -->

## Outcomes

| | Outcomes | Level | Evidence |
|---|---|---|---|
| LO1 | Explain derivatives, partial derivatives, gradients and Jacobians as local linear approximations, given functions of one and several variables | Awareness | supervised AI-off test, at least 70% correct |
| LO2 | Distinguish convex from non-convex objectives and describe how saddle points and step size shape gradient descent, given six loss surfaces with optimisation traces | Awareness | annotated plots marked against a reference |
| LO3 | Explain how a constraint moves the optimum of a small problem, using the Lagrangian and the meaning of its multiplier, given three problems, including a budget allocation | Awareness | written solutions marked against a rubric, with each multiplier read as the price of the constraint |
| LO4 | Compute gradients of a composite function by applying the chain rule backwards through a computational graph, given a two-example logistic-regression loss | Applied | a hand-worked backward pass matching a numerical gradient check |
| LO5 | Implement reverse-mode automatic differentiation for scalar operations and verify it with finite-difference checks, given a starter interface and test suite | Applied | an engine that passes the tests and agrees with a framework's gradients |
| LO6 | Apply batch, stochastic and momentum gradient descent and diagnose divergence or slow convergence from loss curves, given losses on public data | Applied | an experiment log with learning-rate sweeps and a diagnosis of each failed run |

<!-- academy:units -->
## Units

| Unit | Title | Sessions | Slides |
|---|---|---|---|
| Week 1 | Change and approximation | Derivatives as local lines | [`slides/week1.html`](slides/week1.html) |
| Week 2 | Several variables | Gradients and Jacobians | [`slides/week2.html`](slides/week2.html) |
| Week 3 | The chain rule on graphs | Graphs and the chain rule | [`slides/week3.html`](slides/week3.html) |
| Week 4 | Reverse mode in code | A reverse-mode engine | [`slides/week4.html`](slides/week4.html) |
| Week 5 | Gradient descent | Descent and loss curves | [`slides/week5.html`](slides/week5.html) |
| Week 6 | The shape of a loss | Convexity and curvature | [`slides/week6.html`](slides/week6.html) |
| Week 7 | Stable losses and penalties | Stable losses and penalties | [`slides/week7.html`](slides/week7.html) |
| Week 8 | Constraints | Constraints and prices | [`slides/week8.html`](slides/week8.html) |
| Capstone | Capstone oral checks |  | [`slides/capstone.html`](slides/capstone.html) |
<!-- /academy:units -->

## Case study

**The adult learning charity.** A non-profit that runs evening literacy and numeracy classes for
adults in five districts, D1 to D5, of one city. Every charity figure in the labs is synthetic,
generated with a fixed seed by `labs/_tools/make_data.py`: 600 learners coded LR-0001 to LR-0600,
the outreach response curve of each district and the outreach budget. People appear only as
roles: the programme lead, the data analyst and the finance officer. The gradient-descent lab and
the capstone use public datasets instead (see Public data below), never presented as the
charity's.

<!-- academy:labs -->
## Labs

Every lab is a notebook in `labs/`. Open it in Google Colab from the course page, or run it locally with `uv`. Exercises raise `NotImplementedError` until they are done; the reference versions are in `labs/solutions/`.

| | Session | Lab | File |
|---|---|---|---|
| 1.1 | Week 1 · Session 1 | Three ways to a derivative | [`W1_1_derivatives.ipynb`](labs/W1_1_derivatives.ipynb) |
| 2.1 | Week 2 · Session 1 | Gradients and a Jacobian, checked | [`W2_1_gradients.ipynb`](labs/W2_1_gradients.ipynb) |
| 3.1 | Week 3 · Session 1 | Backpropagation by hand | [`W3_1_backprop_by_hand.ipynb`](labs/W3_1_backprop_by_hand.ipynb) |
| 4.1 | Week 4 · Session 1 | A scalar reverse-mode engine | [`W4_1_autodiff_engine.ipynb`](labs/W4_1_autodiff_engine.ipynb) |
| 5.1 | Week 5 · Session 1 | Learning-rate sweeps | [`W5_1_lr_sweeps.ipynb`](labs/W5_1_lr_sweeps.ipynb) |
| 6.1 | Week 6 · Session 1 | Six loss surfaces | [`W6_1_loss_surfaces.ipynb`](labs/W6_1_loss_surfaces.ipynb) |
| 7.1 | Week 7 · Session 1 | Stable softmax and a penalty | [`W7_1_stable_softmax.ipynb`](labs/W7_1_stable_softmax.ipynb) |
| 8.1 | Week 8 · Session 1 | A budget allocation two ways | [`W8_1_budget_allocation.ipynb`](labs/W8_1_budget_allocation.ipynb) |
<!-- /academy:labs -->

The catalog entry lists five labs: 1.1, 3.1, 4.1, 5.1 and 8.1 keep its wording. The format asks for
one lab a week, so three labs are **additions** drawn from the entry's outcomes and topics, not new
topics: 2.1 (gradients and Jacobians, practice for LO1), 6.1 (six loss surfaces, the evidence form
of LO2) and 7.1 (numerically stable softmax and regularisation as a penalty).

## Capstone

A two-layer network trained on public data with the participant's own engine, with gradient checks, three optimisers compared across seeds and every run reported, defended in a ten-minute oral check

**Submission:** a link to a public GitHub repository with the participant's engine, its tests and
gradient checks, the training code, the log of every run with its seed and the report; then a
ten-minute oral check, questions included, in which the participant traces a backward pass through
their engine and explains a failed run from the log. The brief, the rubric and the data loader are
in [`labs/capstone/`](labs/capstone/BRIEF.md).

## Hours

45 hours over 8 weeks and a capstone day.

| Component | Hours | Minutes | What |
|---|---|---|---|
| Instruction | 14 | 840 | 8 sessions × 105 minutes of teaching, one a week |
| Lab | 14 | 840 | 8 labs × 105 minutes, one after each session |
| Guided self-study | 8 | 480 | pre-work refresher 80; Problem sets 1 to 8, 50 each (400) |
| Assessment | 3 | 180 | Test 1 after week 4 and Test 2 after week 8, 90 each: 40 on paper, 50 at the keyboard |
| Capstone | 6 | 360 | brief and planning clinic 30; build 255; gradient check and a practice oral in pairs 30; oral check 10; changeover 5; written feedback 30 |
| **Total** | **45** | **2,700** | |

## Assessment

- Weekly problem sets, auto-checked (20%)
- Two supervised AI-off tests, on paper and at the keyboard (40%)
- Capstone engine and report, rubric-scored (30%)
- Oral check (10%)

Passing: 60% overall and at least 50% on each supervised test

Authenticity: Proctored tests without assistance, and an oral check in which the participant traces a backward pass through their engine and explains a failed run from the log

Certificate: completion with assessed work

The capstone rubric scores the repository and report; the oral check is scored separately, on
understanding shown live, so the same evidence is not counted twice.

## AI policy

AI-off for first attempts at problem sets, the hand backpropagation lab and both tests. AI-tutor
in the other labs. AI-full in the capstone, which is defended orally.

The AI level of each lab is in [`labs/README.md`](labs/README.md); the tutor's hint-only
instruction is in [`labs/TUTOR.md`](labs/TUTOR.md). No lab calls an AI model.

## Responsible use

Covers transparent reporting: the capstone lists every run, seed and failure, not the best result
alone, and the allocation lab discusses how optimising a convenient proxy can harm the people a
model serves.

## Answers and keys

The lab solutions are public, in `labs/solutions/`, with the labs' answer key
(`labs/solutions/answer_key.md`). The problem sets are checked by property in
`labs/problem_sets/check.py`, which holds no worked answers: it compares each function with
finite differences, its own runs or the definition of the quantity asked, so reading it shows
what is checked, not the derivation the trainer marks. Worked problem-set answers, the capstone
reference run, the oral-check sheet and the LO3 marking rubric are kept by the trainer outside
this repository, because publishing them would defeat the assessment.

**The two supervised tests have no papers yet.** Their format is fixed (90 minutes each, 40 on
paper and 50 at the keyboard, AI-off, Test 1 after week 4 on weeks 1 to 4 and Test 2 after week 8),
but the papers and their keys are still to be written, and will be kept off this repository.

## Running the labs

Open a notebook in Google Colab from the course page, or run it locally with
[uv](https://docs.astral.sh/uv/) from `labs/` (see [`labs/README.md`](labs/README.md)). Set
`MTH_QUICK=1` for a short run. Build and check the course with the Academy engine:
`node academy.mjs check`, `node academy.mjs build`.

## Public data

Lab 5.1 uses [Airfoil Self-Noise](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise)
(doi 10.24432/C5VW2C) and the capstone uses
[Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication)
(doi 10.24432/C55P57), both from the UCI Machine Learning Repository under the Creative Commons
Attribution 4.0 licence. The files are downloaded when a lab runs, checked against a pinned
SHA-256 and kept in `labs/data/cache/`, which git ignores; they are not part of this repository.

## Prerequisites

- FND-MTH-101 Linear Algebra for Machine Learning
- School calculus: derivatives of polynomials and exponentials, or the pre-work refresher
- Writes short Python programs with functions and loops (FND-PRG-101 or equivalent)

<!-- academy:tools -->
## Tools

Each line is a capability; the examples are interchangeable.

- **Array computing:** NumPy, JAX
- **Symbolic algebra:** SymPy, Maxima
- **Reference automatic differentiation:** PyTorch autograd, JAX
- **Notebook environment:** JupyterLab, marimo, a free hosted notebook service
- **Hint-only tutor:** an open-weight model in tutor mode, an approved assistant with a tutor instruction
<!-- /academy:tools -->

<!-- academy:sources -->
## Sources

### Differentiation

- [Baydin, Pearlmutter, Radul and Siskind: Automatic differentiation in machine learning: a survey (JMLR 18, 2018)](https://arxiv.org/abs/1502.05767)
- [PyTorch documentation: Autograd mechanics](https://docs.pytorch.org/docs/2.15/notes/autograd.html)
- [SymPy documentation: Calculus (derivatives with diff)](https://docs.sympy.org/latest/tutorials/intro-tutorial/calculus.html)
- [NumPy documentation: numpy.finfo, machine limits for floating-point types](https://numpy.org/doc/stable/reference/generated/numpy.finfo.html)

### Optimisation

- [Ruder: An overview of gradient descent optimization algorithms](https://arxiv.org/abs/1609.04747)
- [Kingma and Ba: Adam: A Method for Stochastic Optimization (ICLR 2015)](https://arxiv.org/abs/1412.6980)
- [Boyd and Vandenberghe: Convex Optimization (Cambridge University Press; free PDF from the authors)](https://web.stanford.edu/~boyd/cvxbook/)

### Public data (CC BY 4.0)

- [Airfoil Self-Noise, UCI Machine Learning Repository, doi 10.24432/C5VW2C](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise)
- [Banknote Authentication, UCI Machine Learning Repository, doi 10.24432/C55P57](https://archive.ics.uci.edu/dataset/267/banknote+authentication)
<!-- /academy:sources -->
