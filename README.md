<!-- academy:title -->
# Calculus and Optimisation for Machine Learning

التفاضل والتكامل والأمثلة الرياضية لتعلم الآلة

Data & AI Foundations Track · SDAIA Academy · FND-MTH-102 · Level 100

REPLACE: one line under the title: what participants can do after the course

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
| Week 1 | REPLACE: the theme of this unit, in a few words | REPLACE: the title of Session 1 | [`slides/week1.html`](slides/week1.html) |
| Week 2 | REPLACE: the theme of this unit, in a few words | REPLACE: the title of Session 1 | [`slides/week2.html`](slides/week2.html) |
| Week 3 | REPLACE: the theme of this unit, in a few words | REPLACE: the title of Session 1 | [`slides/week3.html`](slides/week3.html) |
| Week 4 | REPLACE: the theme of this unit, in a few words | REPLACE: the title of Session 1 | [`slides/week4.html`](slides/week4.html) |
| Week 5 | REPLACE: the theme of this unit, in a few words | REPLACE: the title of Session 1 | [`slides/week5.html`](slides/week5.html) |
| Week 6 | REPLACE: the theme of this unit, in a few words | REPLACE: the title of Session 1 | [`slides/week6.html`](slides/week6.html) |
| Week 7 | REPLACE: the theme of this unit, in a few words | REPLACE: the title of Session 1 | [`slides/week7.html`](slides/week7.html) |
| Week 8 | REPLACE: the theme of this unit, in a few words | REPLACE: the title of Session 1 | [`slides/week8.html`](slides/week8.html) |
| Capstone | Capstone project presentations |  | [`slides/capstone.html`](slides/capstone.html) |
<!-- /academy:units -->

## Case study

**REPLACE: the name of the fictional organisation every lab works on.** REPLACE: two sentences on the case study and its synthetic data

<!-- academy:labs -->
## Labs

Every lab is a notebook in `labs/`. Open it in Google Colab from the course page, or run it locally with `uv`. Exercises raise `NotImplementedError` until they are done; the reference versions are in `labs/solutions/`.

| | Session | Lab | File |
|---|---|---|---|
| 1.1 | Week 1 · Session 1 | REPLACE: what participants build in lab 1.1 | [`W1_1.ipynb`](labs/W1_1.ipynb) |
| 2.1 | Week 2 · Session 1 | REPLACE: what participants build in lab 2.1 | [`W2_1.ipynb`](labs/W2_1.ipynb) |
| 3.1 | Week 3 · Session 1 | REPLACE: what participants build in lab 3.1 | [`W3_1.ipynb`](labs/W3_1.ipynb) |
| 4.1 | Week 4 · Session 1 | REPLACE: what participants build in lab 4.1 | [`W4_1.ipynb`](labs/W4_1.ipynb) |
| 5.1 | Week 5 · Session 1 | REPLACE: what participants build in lab 5.1 | [`W5_1.ipynb`](labs/W5_1.ipynb) |
| 6.1 | Week 6 · Session 1 | REPLACE: what participants build in lab 6.1 | [`W6_1.ipynb`](labs/W6_1.ipynb) |
| 7.1 | Week 7 · Session 1 | REPLACE: what participants build in lab 7.1 | [`W7_1.ipynb`](labs/W7_1.ipynb) |
| 8.1 | Week 8 · Session 1 | REPLACE: what participants build in lab 8.1 | [`W8_1.ipynb`](labs/W8_1.ipynb) |
<!-- /academy:labs -->

## Capstone

A two-layer network trained on public data with the participant's own engine, with gradient checks, three optimisers compared across seeds and every run reported, defended in a ten-minute oral check

**Submission:** REPLACE: what participants submit and how: a public repository link, or a shared folder for a course without code

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

- REPLACE: every source the course cites, with its address, in course.json under "sources"; this list is written from there
<!-- /academy:sources -->
