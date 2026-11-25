---
course: "Calculus and Optimisation for Machine Learning"
code: "FND-MTH-102"
version: "2027.1.0"
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/fnd-mth-102/index.html"
---

# Calculus and Optimisation for Machine Learning

التفاضل والتكامل والأمثلة الرياضية لتعلم الآلة

Data & AI Foundations Track · 8 weeks + capstone · 45 hours · Level 100

Differentiate by hand and in code, build a small reverse-mode engine, and train models with gradient descent you can diagnose

Participants work through the calculus that trains models: derivatives as local linear approximations, gradients, Jacobians and the chain rule on computational graphs. They backpropagate by hand, then write a small reverse-mode automatic differentiation engine and check it against a framework. They run batch, stochastic and momentum gradient descent, read loss curves, and handle convexity and constraints. The capstone trains a small network with their own engine; tensor-level differentiation and modern optimisers follow in SPC-DLR-302.

**Who it is for:** Beginners. Career changers and non-computing graduates closing bridging gaps before advanced study, and computing graduates refreshing the mathematics behind models; a placement test lets participants skip any course whose content they already command.

**Format:** Extended: 8 weeks part-time, one session and one lab a week, weekly graded problem sets and a capstone

## The case study

**The adult learning charity.** A non-profit that runs evening literacy and numeracy classes for adults in five districts, D1 to D5, of one city; its own labs use synthetic data generated with a fixed seed: 600 learners coded LR-0001 to LR-0600, five district outreach response curves and a budget, with planted problems (a wrong hand-coded derivative, logits that overflow, separable placement data and districts that a proxy objective starves). The gradient-descent lab and the capstone train on public datasets from the UCI Machine Learning Repository (CC BY 4.0), downloaded when the lab runs and never presented as the charity's data; people appear only as roles.

## The course

### Week 1: Change and approximation

Slides: https://sdaiaacademy.github.io/fnd-mth-102/slides/week1.md

- **Session 1: Derivatives as local lines**
  - Explain a derivative as the slope of the best local linear approximation
  - Compute derivatives by rule, with symbolic algebra and with finite differences
  - Choose a finite-difference step by weighing truncation against rounding error
  - [Lab 1.1: Three ways to a derivative (105 min)](https://sdaiaacademy.github.io/fnd-mth-102/labs/W1_1_derivatives.ipynb)

### Week 2: Several variables

Slides: https://sdaiaacademy.github.io/fnd-mth-102/slides/week2.md

- **Session 1: Gradients and Jacobians**
  - Compute partial derivatives and assemble the gradient of a function of several variables
  - Explain the gradient as the direction of steepest increase and a Jacobian as the best local linear map
  - Check a hand-derived gradient or Jacobian with central differences and a relative-error tolerance
  - [Lab 2.1: Gradients and a Jacobian, checked (105 min)](https://sdaiaacademy.github.io/fnd-mth-102/labs/W2_1_gradients.ipynb)

### Week 3: The chain rule on graphs

Slides: https://sdaiaacademy.github.io/fnd-mth-102/slides/week3.md

- **Session 1: Graphs and the chain rule**
  - Draw the computational graph of a composite function
  - Explain why reverse mode costs one backward pass per output and forward mode one pass per input
  - Compute a backward pass by hand and confirm it with a numerical gradient check
  - [Lab 3.1: Backpropagation by hand (105 min)](https://sdaiaacademy.github.io/fnd-mth-102/labs/W3_1_backprop_by_hand.ipynb)

### Week 4: Reverse mode in code

Slides: https://sdaiaacademy.github.io/fnd-mth-102/slides/week4.md

- **Session 1: A reverse-mode engine**
  - Apply the chain rule to vector functions as products of Jacobians
  - Design a scalar reverse-mode engine: nodes, local derivatives, topological order and accumulation
  - Verify an engine with finite-difference checks and against a framework's gradients
  - [Lab 4.1: A scalar reverse-mode engine (105 min)](https://sdaiaacademy.github.io/fnd-mth-102/labs/W4_1_autodiff_engine.ipynb)

### Week 5: Gradient descent

Slides: https://sdaiaacademy.github.io/fnd-mth-102/slides/week5.md

- **Session 1: Descent and loss curves**
  - Apply batch, mini-batch stochastic and momentum gradient descent
  - Describe what adaptive step methods, warm-up and decay change
  - Diagnose divergence, oscillation and slow convergence from a loss curve
  - [Lab 5.1: Learning-rate sweeps (105 min)](https://sdaiaacademy.github.io/fnd-mth-102/labs/W5_1_lr_sweeps.ipynb)

### Week 6: The shape of a loss

Slides: https://sdaiaacademy.github.io/fnd-mth-102/slides/week6.md

- **Session 1: Convexity and curvature**
  - Distinguish convex from non-convex objectives with a test you can run
  - Classify critical points from the eigenvalues of the Hessian
  - Relate the largest stable step to the largest curvature
  - [Lab 6.1: Six loss surfaces (105 min)](https://sdaiaacademy.github.io/fnd-mth-102/labs/W6_1_loss_surfaces.ipynb)

### Week 7: Stable losses and penalties

Slides: https://sdaiaacademy.github.io/fnd-mth-102/slides/week7.md

- **Session 1: Stable losses and penalties**
  - Compute softmax and log-sum-exp without overflow
  - Derive and check the gradient of softmax cross-entropy
  - Explain regularisation as a penalty that reshapes the loss and bounds the weights
  - [Lab 7.1: Stable softmax and a penalty (105 min)](https://sdaiaacademy.github.io/fnd-mth-102/labs/W7_1_stable_softmax.ipynb)

### Week 8: Constraints

Slides: https://sdaiaacademy.github.io/fnd-mth-102/slides/week8.md

- **Session 1: Constraints and prices**
  - Solve a small equality-constrained problem with the Lagrangian
  - Read a multiplier as the price of its constraint
  - Explain how optimising a convenient proxy can harm the people a model serves
  - [Lab 8.1: A budget allocation two ways (105 min)](https://sdaiaacademy.github.io/fnd-mth-102/labs/W8_1_budget_allocation.ipynb)

### Capstone: Capstone oral checks

Slides: https://sdaiaacademy.github.io/fnd-mth-102/slides/capstone.md

## Capstone

A two-layer network trained on public data with the participant's own engine, with gradient checks, three optimisers compared across seeds and every run reported, defended in a ten-minute oral check

**What you submit:** A link to a public GitHub repository with the participant's engine, its tests and gradient checks, the training code, the log of every run with its seed and the report; then a ten-minute oral check, questions included, in which the participant traces a backward pass through their engine and explains a failed run from the log

**Rubric:**

- Engine correct: passes the course test suite and the gradient checks on the network: 25%
- Network trained on the public data with a held-out split, reproducible from the seeds: 20%
- Three optimisers compared across at least three seeds, with loss curves and held-out accuracy: 25%
- Every run, seed and failure reported, not the best result alone: 15%
- A written diagnosis of each failed run from its loss curve and log: 15%

## Outcomes

- **LO1** Explain derivatives, partial derivatives, gradients and Jacobians as local linear approximations, given functions of one and several variables (Awareness) Evidence: supervised AI-off test, at least 70% correct
- **LO2** Distinguish convex from non-convex objectives and describe how saddle points and step size shape gradient descent, given six loss surfaces with optimisation traces (Awareness) Evidence: annotated plots marked against a reference
- **LO3** Explain how a constraint moves the optimum of a small problem, using the Lagrangian and the meaning of its multiplier, given three problems, including a budget allocation (Awareness) Evidence: written solutions marked against a rubric, with each multiplier read as the price of the constraint
- **LO4** Compute gradients of a composite function by applying the chain rule backwards through a computational graph, given a two-example logistic-regression loss (Applied) Evidence: a hand-worked backward pass matching a numerical gradient check
- **LO5** Implement reverse-mode automatic differentiation for scalar operations and verify it with finite-difference checks, given a starter interface and test suite (Applied) Evidence: an engine that passes the tests and agrees with a framework's gradients
- **LO6** Apply batch, stochastic and momentum gradient descent and diagnose divergence or slow convergence from loss curves, given losses on public data (Applied) Evidence: an experiment log with learning-rate sweeps and a diagnosis of each failed run

## Tools

- **Array computing:** NumPy, JAX
- **Symbolic algebra:** SymPy, Maxima
- **Reference automatic differentiation:** PyTorch autograd, JAX
- **Notebook environment:** JupyterLab, marimo, a free hosted notebook service
- **Hint-only tutor:** an open-weight model in tutor mode, an approved assistant with a tutor instruction

## Prerequisites

- FND-MTH-101 Linear Algebra for Machine Learning
- School calculus: derivatives of polynomials and exponentials, or the pre-work refresher
- Writes short Python programs with functions and loops (FND-PRG-101 or equivalent)

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

---

SDAIA Academy · Data & AI Foundations Track · Calculus and Optimisation for Machine Learning

Course version 2027.1.0 · engine 1.0.0 · built 2026-11-25 · commit 413d7b3
