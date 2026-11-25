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

**REPLACE: the name of the fictional organisation every lab works on.** REPLACE: two sentences on the case study and its synthetic data

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
