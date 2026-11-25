---
course: "Calculus and Optimisation for Machine Learning"
code: "FND-MTH-102"
version: "2027.1.0"
scope: "course"
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/fnd-mth-102/"
---

# Calculus and Optimisation for Machine Learning

Differentiate by hand and in code, build a small reverse-mode engine, and train models with gradient descent you can diagnose

Participants work through the calculus that trains models: derivatives as local linear approximations, gradients, Jacobians and the chain rule on computational graphs. They backpropagate by hand, then write a small reverse-mode automatic differentiation engine and check it against a framework. They run batch, stochastic and momentum gradient descent, read loss curves, and handle convexity and constraints. The capstone trains a small network with their own engine; tensor-level differentiation and modern optimisers follow in SPC-DLR-302.

- **Audience:** Beginners. Career changers and non-computing graduates closing bridging gaps before advanced study, and computing graduates refreshing the mathematics behind models; a placement test lets participants skip any course whose content they already command.
- **Prerequisites:** FND-MTH-101 Linear Algebra for Machine Learning; School calculus: derivatives of polynomials and exponentials, or the pre-work refresher; Writes short Python programs with functions and loops (FND-PRG-101 or equivalent)
- **Case study:** The adult learning charity: A non-profit that runs evening literacy and numeracy classes for adults in five districts, D1 to D5, of one city; its own labs use synthetic data generated with a fixed seed: 600 learners coded LR-0001 to LR-0600, five district outreach response curves and a budget, with planted problems (a wrong hand-coded derivative, logits that overflow, separable placement data and districts that a proxy objective starves). The gradient-descent lab and the capstone train on public datasets from the UCI Machine Learning Repository (CC BY 4.0), downloaded when the lab runs and never presented as the charity's data; people appear only as roles.

## Outcomes

- LO1 (Awareness): Explain derivatives, partial derivatives, gradients and Jacobians as local linear approximations, given functions of one and several variables
- LO2 (Awareness): Distinguish convex from non-convex objectives and describe how saddle points and step size shape gradient descent, given six loss surfaces with optimisation traces
- LO3 (Awareness): Explain how a constraint moves the optimum of a small problem, using the Lagrangian and the meaning of its multiplier, given three problems, including a budget allocation
- LO4 (Applied): Compute gradients of a composite function by applying the chain rule backwards through a computational graph, given a two-example logistic-regression loss
- LO5 (Applied): Implement reverse-mode automatic differentiation for scalar operations and verify it with finite-difference checks, given a starter interface and test suite
- LO6 (Applied): Apply batch, stochastic and momentum gradient descent and diagnose divergence or slow convergence from loss curves, given losses on public data

## Week 1: Change and approximation

Derivatives as local linear approximations, and three ways to compute one

#### [week1#title] Change and approximation

*SDAIA Academy · Calculus and Optimisation for Machine Learning*

Week 1: Derivatives as local linear approximations, and three ways to compute one

Data & AI Foundations Track · 8 weeks + capstone · 1 session · 1 lab · FND-MTH-102

#### [week1#eight-weeks] Eight weeks from a single slope to a network you train with your own engine

*The course at a glance*

- **1 · Change.** Derivatives, Taylor, finite differences
- **2 · Several variables.** Gradients and Jacobians
- **3 · Graphs.** The chain rule, backprop by hand
- **4 · Reverse mode.** Your own differentiation engine
- **5 · Descent.** Step size, batches, momentum, loss curves
- **6 · Shape.** Convexity, saddles, curvature
- **7 · Stability.** Log-sum-exp, softmax, penalties
- **8 · Constraints.** Lagrange multipliers, projections

> **Capstone.** A two-layer network on public data, trained with the engine you write in week 4, defended in a ten-minute oral check.

#### [week1#how-assessed] Four parts, and two tests you sit without help

*How the course is assessed*

| Item | Value |
| --- | --- |
| Weekly problem sets, auto-checked | 20% |
| Two supervised AI-off tests | 40% |
| Capstone engine and report | 30% |
| Oral check | 10% |

- **To pass.** 60% overall and at least 50% on each supervised test.
- **AI levels.** **Off**: first attempts at problem sets, Lab 3.1, both tests. **Tutor**: the other labs. **Full**: the capstone, defended orally.

#### [week1#the-charity] The adult learning charity: evening classes in five districts

*One case all term*

A non-profit that teaches adults to read, write and calculate. Its data analyst wants to understand the models before trusting them.

- **Completion model.** The chance a learner finishes, from attendance and weekly hours. Weeks 1 to 3.
- **Placement model.** Three class levels from an entry score. Week 7.
- **Outreach budget.** Enrolment curves for districts D1 to D5, one budget. Weeks 1, 2 and 8.

> **All synthetic.** 600 learners coded LR-0001 onwards, generated with a fixed seed. Week 5 and the capstone use public datasets instead.

#### [week1#setup-check] Open the setup check and run it once

*Before the first lab*

1. Colab: open `00_setup_check.ipynb` from the course page and run every cell.
2. Laptop: clone the repository, then run the commands on the right from `labs/`.
3. It lists missing libraries and downloads the two public datasets once.
4. No lab calls an AI model: no key, no account.

```bash
# file: terminal
cd fnd-mth-102/labs
uv sync
uv run python _tools/py2nb.py
uv run jupyter lab 00_setup_check.ipynb
```

#### [week1#agenda] Change and approximation

*This week*

- **Session 1 · Derivatives as local lines**: A derivative is the slope of the best straight-line fit near a point; you compute it by hand, symbolically and numerically

### Session 1: Derivatives as local lines

#### [week1#s1] Derivatives as local lines

*Session 1*

- Explain a derivative as the slope of the best local linear approximation
- Compute derivatives by rule, with symbolic algebra and with finite differences
- Choose a finite-difference step by weighing truncation against rounding error

#### [week1#secant-to-tangent] A derivative is the slope a secant settles on as the gap closes

*Block 1 · What a derivative is*

The slope between `x` and `x + h` is `(f(x + h) - f(x)) / h`.

Shrink `h`: for a smooth function the slope settles on one number, **f′(x)**.

| h | slope of x² at 3 |
| --- | --- |
| 1 | 7 |
| 0.1 | 6.1 |
| 0.01 | 6.01 |
| limit | 6 |

- **Read it as a rate.** f′(x) is how fast `f` changes per unit of `x`, right at `x`: enrolments per extra thousand of budget, loss per unit change of a weight.

#### [week1#best-linear-fit] The derivative gives the best straight-line fit near a point

*Block 1 · The definition that generalises*

- **Local linear approximation.** `f(x + h) ≈ f(x) + f′(x) · h`

The error shrinks **faster than h**: halve the step and the error falls to about a quarter.

No other straight line through `(x, f(x))` does that. This, not "slope of a curve", is the definition that carries to many variables in week 2.

- **Why machine learning cares.** Training changes weights by small steps. Near the current weights, the loss changes by about **derivative × step**. Every optimiser in weeks 5 to 8 starts from that line.

#### [week1#tangent-d3] District D3's enrolment curve and its tangent at 12 thousand

*Block 1 · In the case*

Figure: Synthetic: E(x) = 45 · ln(1 + x / 8), drawn by labs/_tools/make_figures.py.

**2.25** enrolments per extra thousand at x = 12

E′(x) = 45 / (8 + x), so E′(12) = 45 / 20.

The tangent says 2 more thousand bring 4.5 enrolments; the curve gives 4.3.

#### [week1#check-local-estimate] What does the slope promise the finance officer?

*Checkpoint 1*

**Question:** District D2's curve has E(15) = 50 enrolments and E′(15) = 1.8 per thousand. The finance officer asks what 2 more thousand of outreach would bring. Which answer is best?

- A. About 1.8 more enrolments
- B. About 3.6 more, slightly fewer since the curve flattens
- C. Exactly 3.6 more enrolments
- D. About 90 more enrolments

#### [week1#rules-table] Five rules cover every function in this course

*Block 2 · Computing derivatives exactly*

| Rule | Statement | Example |
| --- | --- | --- |
| Power | `(xⁿ)′ = n xⁿ⁻¹` | `(x³)′ = 3x²` |
| Exponential and log | `(eˣ)′ = eˣ`, `(ln x)′ = 1/x` | `(e^(0.5x))′ = 0.5 e^(0.5x)` |
| Sum and constant | `(a f + b g)′ = a f′ + b g′` | `(4x² - x)′ = 8x - 1` |
| Product | `(f g)′ = f′ g + f g′` | `(x eˣ)′ = (1 + x) eˣ` |
| Chain | `f(g(x))′ = f′(g(x)) · g′(x)` | `ln(1 + x/8)′ = 1 / (8 + x)` |

> **Sigmoid.** σ(z) = 1 / (1 + e⁻ᶻ) has σ′(z) = σ(z)(1 - σ(z)): you will use it every week.

#### [week1#symbolic-algebra] Symbolic algebra applies the same rules exactly, and can bloat

*Block 2 · Rules, done by a program*

```python
# file: symbolic.py
import sympy as sp

x = sp.symbols("x", real=True)
g = sp.exp(sp.Rational(1, 2) * x) * x**2
dg = sp.diff(g, x)
print(sp.factor(dg))   # x*(x + 4)*exp(x/2)/2
f = sp.lambdify(x, dg, "numpy")
print(f(1.0))          # 4.1218...
```

- **Exact.** No step size, no rounding until you evaluate. Good for checking a hand derivation.
- **Expression swell.** Differentiate a long composition symbolically and the formula can grow far larger than the program that computes it. Reverse mode (week 4) avoids that.

#### [week1#taylor-quadratic] Add the curvature term and the approximation reaches further

*Block 2 · Taylor expansions*

- **Taylor, to second order.** `f(x + h) ≈ f(x) + f′(x) h + ½ f″(x) h²`

The first neglected term is of order `h³`. The sign of `f″` says which side of the tangent the curve lies on.

**f″ < 0** (concave): the tangent overestimates. **f″ > 0** (convex): it underestimates.

| Extra budget, D3 from 12 | Tangent off by |
| --- | --- |
| 2 thousand | 0.5% |
| 5 thousand | 2.4% |
| 10 thousand | 7.2% |
| 20 thousand | 19.1% |

#### [week1#check-concave-tangent] Which side of the tangent is the curve on?

*Checkpoint 2*

**Question:** A completion-rate curve r(h) of weekly hours h has r″(h) < 0 for every h. The analyst uses the tangent line at h = 3 to predict r at h = 6. What can you say before computing anything?

- A. The prediction is too low
- B. The prediction is exact
- C. It depends on which side of 3 we predict
- D. The prediction is too high

#### [week1#finite-differences] Finite differences need only the function, and pay with error

*Block 3 · Derivatives from values alone*

- **Forward.** `(f(x + h) - f(x)) / h` Truncation error about `h/2 · |f″|`: first order.
- **Central.** `(f(x + h) - f(x - h)) / (2h)` The h² terms cancel: error about `h²/6 · |f‴|`, second order.

- **Their real job.** Nobody trains a model with finite differences: one extra evaluation per weight is far too slow. They are the **referee**: every gradient you derive or code in this course is checked against a central difference.

#### [week1#rounding-error] A float holds about 16 digits, and subtraction throws them away

*Block 3 · The second error*

- Machine epsilon, 64-bit: **2.2 × 10⁻¹⁶**, the relative gap between neighbouring floats.
- For tiny `h`, `f(x + h)` and `f(x)` share most digits; their difference keeps the noisy ones.
- Dividing by `h` magnifies the noise: rounding error ≈ `ε |f| / h`.

```python
# file: python
>>> import numpy as np
>>> np.finfo(float).eps
2.220446049250313e-16
>>> (1.0 + 1e-16) - 1.0
0.0
```

> **What goes wrong.** A gradient check with `h = 1e-14` "fails" for a correct gradient. The check is broken, not the gradient.

#### [week1#error-v] Error falls with h, then rises: the best step sits in the V

*Block 3 · Two errors, one best step*

Figure: g(x) = e^(0.5x) at x = 1, steps from 0.1 to 1e-12; labs/_tools/make_figures.py.

|  | Theory | This example |
| --- | --- | --- |
| Best forward step | √ε ≈ 1.5e-8 | 3e-8 |
| Best central step | ∛ε ≈ 6.1e-6 | 3e-6 |
| Lowest error, forward |  | 6e-9 |
| Lowest error, central |  | 5e-13 |

#### [week1#check-step-size] Which step should the gradient check use next?

*Checkpoint 3*

**Question:** A colleague checks a hand-coded derivative with a forward difference. With h = 1e-14 the mismatch is 1e-2; with h = 1e-3 it is 1e-4. She concludes the derivative is wrong. Which step should she try first?

- A. A forward step near 1e-8
- B. A forward step of 1e-16
- C. A forward step of 1e-1
- D. A central step of 1e-14

#### [week1#lab-1-1] Three ways to a derivative

Lab 1.1 · 105 min

Analytic, symbolic and finite-difference derivatives compared across step sizes

[W1_1_derivatives.ipynb](https://sdaiaacademy.github.io/fnd-mth-102/labs/W1_1_derivatives.ipynb)

- [ ] Derive three derivatives by hand and code them (15 min)
- [ ] Get the same derivatives from SymPy and compare (15 min)
- [ ] Write forward and central differences (15 min)
- [ ] Sweep the step from 0.1 to 1e-12 and plot the error (20 min)
- [ ] Explain the V: truncation against rounding, and the best step (15 min)
- [ ] Test how far the tangent line and the quadratic hold (15 min)
- [ ] Write three findings in the lab log (10 min)
- [ ] ✱ Stretch: Compute the complex-step derivative and see why it has no rounding V

#### [week1#problem-set-1] Problem set 1: six derivatives and one explanation

*Before week 2*

- **50 minutes**, graded, auto-checked: `labs/problem_sets/PS1.py`.
- **AI-off for the first attempt.** Work on paper, then code your answers.
- The checker tests your functions against finite differences; it holds no answer key.
- Hand in the file before the week 2 session.

```bash
# file: terminal
cd labs
uv run python problem_sets/check.py PS1
```

#### [week1#wrap-up] Three things to keep

*Week 1 · wrap-up*

- **A local line.** f(x + h) ≈ f(x) + f′(x) h, with an error that shrinks faster than h.
- **Curvature decides the side.** The second-order term says how far the line holds and whether it over- or underestimates.
- **Finite differences referee.** Central differences with a step near 1e-5 check any derivative; too small a step breaks the check.

#### [week1#next] Week 2: Several variables

*Next*

Hand in Problem set 1. Finish Lab 1.1 at home if needed: week 2 opens with its results.

## Week 2: Several variables

Partial derivatives, gradients and Jacobians, and how to check them

#### [week2#title] Several variables

*SDAIA Academy · Calculus and Optimisation for Machine Learning*

Week 2: Partial derivatives, gradients and Jacobians, and how to check them

Data & AI Foundations Track · 8 weeks + capstone · 1 session · 1 lab · FND-MTH-102

#### [week2#agenda] Several variables

*This week*

- **Session 1 · Gradients and Jacobians**: Hold all inputs but one still to get a partial derivative; stack them into a gradient, or into a Jacobian for a map with several outputs

#### [week2#review-week1] How does the central difference's error change when you halve h?

*Review of week 1*

**Question:** With h = 0.01, the central difference of a smooth function is off by 4 × 10⁻⁵. You halve the step to 0.005, still far above the rounding region. About how large is the error now?

- A. 2 × 10⁻⁵
- B. 4 × 10⁻⁵
- C. 1 × 10⁻⁵
- D. 8 × 10⁻⁵

#### [week2#lab-1-1-debrief] Theory predicted the best steps within a factor of five

*Lab 1.1 · what your runs showed*

| From the reference run | Value |
| --- | --- |
| Hand against SymPy, largest difference | 4e-16 |
| Best central step, measured (theory) | 1e-5 (6e-6) |
| Best forward step, measured (theory) | 7e-8 (1.5e-8) |
| Tangent of D1 at 20 within 1% up to | 5 thousand |
| Quadratic within 1% up to | 10 thousand |

- **E′(20) = 2.0 exactly.** 60 / (10 + 20): two enrolments per extra thousand, for small changes only.
- **At 5 thousand, just.** The tangent's error at a 5-thousand change is 0.999%: it sits on the edge.

### Session 1: Gradients and Jacobians

#### [week2#s1] Gradients and Jacobians

*Session 1*

- Compute partial derivatives and assemble the gradient of a function of several variables
- Explain the gradient as the direction of steepest increase and a Jacobian as the best local linear map
- Check a hand-derived gradient or Jacobian with central differences and a relative-error tolerance

#### [week2#partial-derivative] A partial derivative freezes every input but one

*Block 1 · Partial derivatives*

For `f(u, v)`, the partial **∂f/∂u** is the week 1 derivative of `u ↦ f(u, v)` with `v` held fixed.

Geometrically: slice the surface along the u-direction and take the slope of the slice.

- **Example.** `f(u, v) = u² v + e^v`
  `∂f/∂u = 2uv`, `∂f/∂v = u² + e^v`

- **What goes wrong.** Forgetting that the frozen input is still in the formula. In `∂f/∂u = 2uv`, the value of v matters: the slope along u differs from slice to slice.

#### [week2#partials-in-a-model] In a logistic model every partial shares the factor σ(1 - σ)

*Block 1 · In a model*

```python
# file: re-enrolment model (parallel example)
r(u, v) = sigmoid(3*u + 2*v - 2.5)

dr/du = 3 * r * (1 - r)
dr/dv = 2 * r * (1 - r)

# at (u, v) = (0.45, 0.45): r = 0.438
# dr/du = 0.738, dr/dv = 0.492
```

- **The ratio is the weights'.** dr/du ÷ dr/dv = 3 ÷ 2 at every point.
- **The size is the curve's.** σ(1 - σ) is largest, 0.25, where r = 0.5, and near 0 when the model is sure. Sure models learn slowly: week 7.

#### [week2#check-partials] Which partial derivative follows?

*Checkpoint 1*

**Question:** A model p = σ(3a + 0.4h - 2) gives a learner p = 0.5, and the analyst computes ∂p/∂h = 0.1 for that learner. What is ∂p/∂a for the same learner?

- A. 0.1
- B. 3
- C. 0.25
- D. 0.75

#### [week2#gradient-vector] The gradient turns week 1's line into a plane

*Block 2 · The gradient*

- **Local linear approximation, n inputs.** `f(x + δ) ≈ f(x) + ∇f(x) · δ`

`∇f = (∂f/∂x₁, …, ∂f/∂xₙ)`: one number per input, stacked into a vector.

The dot product adds the effect of each input's change, as if each acted alone. Near `x` that is accurate.

- **Directional derivative.** The rate of change along a unit vector `u` is `∇f · u`. For the re-enrolment model at (0.45, 0.45): along u alone 0.738; along (0.6, 0.8) it is 0.837.

#### [week2#steepest-ascent] The gradient points uphill, across the level sets

*Block 2 · Direction of steepest increase*

Figure: r(u, v) = σ(3u + 2v - 2.5), a parallel example; labs/_tools/make_figures.py.

`∇f · u = |∇f| cos θ`: largest when `u` points along `∇f`.

The largest rate is **|∇f|**: 0.887 at (0.45, 0.45).

Along a level set `f` does not change, so the gradient is perpendicular to it.

> **Descent.** −∇f is the steepest way down: week 5.

#### [week2#check-steepest] Which small change raises the score most?

*Checkpoint 2*

**Question:** At (0.45, 0.45) the re-enrolment model has gradient (0.738, 0.492). You may change (u, v) by a step of length 0.1. Which step raises r the most?

- A. (0.1, 0)
- B. (0.071, 0.071)
- C. (0.083, 0.055)
- D. (0, 0.1)

#### [week2#jacobian-matrix] A Jacobian stacks one gradient per output

*Block 3 · Several outputs*

For `F: ℝⁿ → ℝᵐ`, the Jacobian **J** is `m × n`: row i is `∇Fᵢ`, entry `(i, j)` is `∂Fᵢ/∂xⱼ`.

- **Local linear map.** `F(x + δ) ≈ F(x) + J δ`

A matrix times a vector, as in FND-MTH-101: J is the best linear map near x.

- **The district map.** Five budgets in, five enrolment counts out. Each district depends only on its own budget, so every off-diagonal entry is 0: **J is diagonal**, with `kᵢ / (sᵢ + xᵢ)` on the diagonal.

#### [week2#jacobian-predicts] The Jacobian predicts a budget shift well when the shift is small

*Block 3 · Using a Jacobian*

| Split (40, 30, 20, 15, 15); move from D2 to D5 | Jacobian predicts | Model gives |
| --- | --- | --- |
| 5 thousand | -4.19 | -4.72 |
| 15 thousand | -12.57 | -17.98 |

Diagonal of J at this split: 1.20, 1.24, 1.61, 0.60, 0.40 enrolments per thousand.

> **What goes wrong.** Trusting a linear prediction for a large move. Both curves bend, so the error grows faster than the move.

#### [week2#gradient-checking] Check every coordinate with central differences and a relative error

*Block 3 · Checking a gradient*

```python
# file: check.py
def rel_err(g, g_num):
    return norm(g - g_num) / max(norm(g), norm(g_num))

# g_num[i] = (f(x + h e_i) - f(x - h e_i)) / (2h)
# h = 1e-6, one pair of evaluations per input
```

| Relative error | Read it as |
| --- | --- |
| below 1e-7 | correct |
| 1e-7 to 1e-2 | suspicious: a kink, a large step, or a small slip |
| above 1e-2 | a wrong formula |

> **Look per coordinate.** One wrong entry among correct ones points at one formula.

#### [week2#check-gradient-reading] What does this gradient check say?

*Checkpoint 3*

**Question:** A three-parameter gradient is checked with central differences, h = 1e-6. The relative errors of the components are 2e-10, 0.4 and 3e-10. What is the most likely cause?

- A. The formula for the second component is wrong
- B. The step h is too small for float64
- C. The gradient is fine; the errors are rounding
- D. Central differences cannot check this function

#### [week2#lab-2-1] Gradients and a Jacobian, checked

Lab 2.1 · 105 min

Partial derivatives, a gradient, two Jacobians and a reusable gradient checker on the charity's completion and enrolment models

[W2_1_gradients.ipynb](https://sdaiaacademy.github.io/fnd-mth-102/labs/W2_1_gradients.ipynb)

- [ ] Derive the partial derivatives of the completion model (15 min)
- [ ] Build the gradient and test the steepest direction (15 min)
- [ ] Write a gradient checker for any function of n inputs (15 min)
- [ ] Build two Jacobians and check them (20 min)
- [ ] Predict a budget shift with the Jacobian and compare (15 min)
- [ ] Find the planted wrong partial derivative (15 min)
- [ ] Write three findings in the lab log (10 min)
- [ ] ✱ Stretch: Check the symmetry of second partial derivatives numerically

#### [week2#problem-set-2] Problem set 2: gradients, a Jacobian and a direction

*Before week 3*

- **50 minutes**, graded, auto-checked: `labs/problem_sets/PS2.py`.
- **AI-off for the first attempt.** Paper first, then code.
- Two gradients, a directional derivative, a 3 × 2 Jacobian, a steepest-descent direction, one written answer.

```bash
# file: terminal
cd labs
uv run python problem_sets/check.py PS2
```

#### [week2#wrap-up] Three things to keep

*Week 2 · wrap-up*

- **Gradient as a plane.** f(x + δ) ≈ f(x) + ∇f · δ: one partial per input, added up.
- **Uphill, across level sets.** The gradient is the steepest direction and its length the steepest rate; −∇f points down.
- **Jacobian as a map.** One row per output; checked column by column with central differences and a relative error.

#### [week2#next] Week 3: The chain rule on graphs

*Next*

Hand in Problem set 2. Bring paper and a pencil: Lab 3.1 is done by hand, AI-off.

## Week 3: The chain rule on graphs

Computational graphs, forward and reverse mode, and a backward pass by hand

#### [week3#title] The chain rule on graphs

*SDAIA Academy · Calculus and Optimisation for Machine Learning*

Week 3: Computational graphs, forward and reverse mode, and a backward pass by hand

Data & AI Foundations Track · 8 weeks + capstone · 1 session · 1 lab · FND-MTH-102

#### [week3#agenda] The chain rule on graphs

*This week*

- **Session 1 · Graphs and the chain rule**: Write a loss as a graph of simple operations; the chain rule run backwards gives every gradient in one pass

#### [week3#review-week2] What shape is this Jacobian?

*Review of week 2*

**Question:** A map takes a learner's four features and returns two scores: the chance to complete and the expected hours attended. What is the shape of its Jacobian?

- A. 4 × 2
- B. 2 × 4
- C. 4 × 4
- D. 2 × 2

#### [week3#lab-2-1-debrief] The Jacobian was right for 5 thousand and badly wrong for 20

*Lab 2.1 · what your runs showed*

| Move from D1 to D4 | Jacobian predicts | Model gives |
| --- | --- | --- |
| 5 thousand | -4.93 | -5.61 |
| 20 thousand | -19.71 | -33.45 |

- **Steepest direction.** 4 degrees from the attendance axis, rate 0.858: the gradient's angle and length.
- **Last term's notes.** The partial for w_h carried an extra factor a: relative error 0.29, caught at once.

### Session 1: Graphs and the chain rule

#### [week3#s1] Graphs and the chain rule

*Session 1*

- Draw the computational graph of a composite function
- Explain why reverse mode costs one backward pass per output and forward mode one pass per input
- Compute a backward pass by hand and confirm it with a numerical gradient check

#### [week3#graph-of-operations] Any formula a program computes is a graph of simple operations

*Block 1 · Computational graphs*

`f(x, y) = (x + y) · y`

1. **inputs** x = 2, y = 3
2. **a = x + y** a = 5
3. **f = a · y** f = 15

- One **node** per operation, an **edge** from each input to its use.
- Each node knows only its **local derivatives**: for `a · y`, they are `y` and `a`.
- y is used **twice**: by the sum and by the product.
- The forward pass evaluates nodes in an order where inputs come first.

#### [week3#chain-rule-on-a-chain] Along a chain, multiply the local derivatives

*Block 1 · The chain rule, scalar*

- **A chain.** `x → u = g(x) → f(u)`
  `df/dx = f′(u) · g′(x)`

Example: `f = ln(1 + e^(2x))` at `x = 0`: `u = 1 + e^(2x) = 2`; `df/du = 1/u = 0.5`; `du/dx = 2 e^(2x) = 2`; so `df/dx = 1`.

- **Two paths: add.** When an input reaches the output along several paths, the derivative is the **sum over paths** of the products along each. In `(x + y) · y`, y has two paths.

#### [week3#backward-pass-example] Go down the graph once, passing adjoints to inputs

*Block 1 · A backward pass*

| Node | Value | Local derivative to its inputs | Adjoint ∂f/∂node |
| --- | --- | --- | --- |
| f = a · y | 15 | to a: y = 3; to y: a = 5 | 1 |
| a = x + y | 5 | to x: 1; to y: 1 | 1 · 3 = 3 |
| x | 2 |  | 3 · 1 = 3 |
| y | 3 |  | 1 · 5 + 3 · 1 = 8 |

> **Accumulate.** y's adjoint is the sum of what each user sends it. Overwriting instead of adding is the classic engine bug.

#### [week3#check-reused-node] What is the adjoint of the reused input?

*Checkpoint 1*

**Question:** In g = u · v with u = 2w and v = w + 1, at w = 3 the values are u = 6, v = 4. What is dg/dw?

- A. 8
- B. 6
- C. 14
- D. 48

#### [week3#forward-mode] Forward mode carries one tangent up with the values

*Block 2 · Forward mode*

Pick one input direction. Each node computes its value **and** its tangent, `d node / d input`, from its inputs' tangents.

| (x + y) · y, tangent along x | Value | Tangent |
| --- | --- | --- |
| x, y | 2, 3 | 1, 0 |
| a = x + y | 5 | 1 |
| f = a · y | 15 | 1 · 3 + 5 · 0 = 3 |

- **One pass, one direction.** A forward pass gives the derivative of **every output** along **one input direction**: one column of the Jacobian. For the full gradient of f you need a pass per input.

#### [week3#mode-costs] Reverse mode wins when many inputs feed one output, as a loss does

*Block 2 · Which mode*

| Function | Forward passes | Backward passes |
| --- | --- | --- |
| n inputs, m outputs (full Jacobian) | n | m |
| A loss of a million weights | 1,000,000 | 1 |
| A curve: one input, many outputs | 1 | many |

> **The price of reverse mode.** The backward pass needs every value of the forward pass, so they are kept in memory until it runs. Saved activations are why training needs more memory than prediction.

#### [week3#check-which-mode] Which mode should the analyst use?

*Checkpoint 2*

**Question:** The analyst needs how all 600 learners' predicted completion chances change with the single attendance threshold the charity sets. One input, 600 outputs. Which costs fewer passes?

- A. Reverse mode, one backward pass per learner
- B. Forward mode, one pass along the threshold
- C. Either: they always cost the same
- D. Central differences instead of either mode

#### [week3#logistic-graph] A logistic loss is four nodes per learner

*Block 3 · Backpropagation by hand*

1. **z = w · x + b** the score
2. **p = σ(z)** the chance
3. **ℓ = −[y ln p + (1 − y) ln(1 − p)]** the loss
4. **L = mean of ℓ** over learners

| Edge | Local derivative |
| --- | --- |
| ℓ to p | `−y/p + (1 − y)/(1 − p)` |
| p to z | `p (1 − p)` |
| ℓ to z, the product of the two | `p − y` |
| z to wⱼ, to b | `xⱼ`, `1` |

#### [week3#one-learner-worked] Forward up, backward down: a worked example

*Block 3 · One learner, worked*

| x = (0.5, 1.0), y = 1; w = (0.8, −0.4), b = 0.2 | Value |
| --- | --- |
| z = 0.4 − 0.4 + 0.2 | 0.2 |
| p = σ(0.2) | 0.5498 |
| ℓ = −ln 0.5498 | 0.5981 |

| Backward | Adjoint |
| --- | --- |
| z: p − y | −0.4502 |
| w₁: −0.4502 · 0.5 | −0.2251 |
| w₂: −0.4502 · 1.0 | −0.4502 |
| b: −0.4502 · 1 | −0.4502 |

> **For two learners,** each learner sends its own adjoints and the mean node halves them: a weight's gradient is the average of the learners' pushes.

#### [week3#hand-work-discipline] Do it by hand once, carefully, and check it against numbers

*Block 3 · Why by hand, and how*

- **Four decimals.** Carry four decimals all the way. Rounding early shows up as a relative error near 1e-3.
- **Check as you go.** After the forward pass, the code's values; after the backward pass, a numerical gradient check.
- **AI-off, on purpose.** The tests and the oral check ask you to trace a backward pass alone. This lab is the practice.

#### [week3#check-z-adjoint] What does the learner who left send back?

*Checkpoint 3*

**Question:** A learner did not complete (y = 0) and the model gave p = 0.8. For this learner's loss, what is the adjoint of z?

- A. −0.2
- B. 0.16
- C. 5.0
- D. 0.8

#### [week3#lab-3-1] Backpropagation by hand

Lab 3.1 · 105 min

Backpropagation by hand on a tiny logistic regression, checked in code

[W3_1_backprop_by_hand.ipynb](https://sdaiaacademy.github.io/fnd-mth-102/labs/W3_1_backprop_by_hand.ipynb)

- [ ] Draw the graph of the two-example loss (15 min)
- [ ] Fill in the forward pass by hand (15 min)
- [ ] Run the backward pass by hand (25 min)
- [ ] Check your forward pass in code (10 min)
- [ ] Match a numerical gradient check (15 min)
- [ ] One directional derivative in forward mode (15 min)
- [ ] Write where your values differed, and why (10 min)
- [ ] ✱ Stretch: Add an L2 term and extend the backward pass by hand

#### [week3#problem-set-3] Problem set 3: four backward passes and a cost

*Before week 4*

- **50 minutes**, graded, auto-checked: `labs/problem_sets/PS3.py`.
- **AI-off for the first attempt.** Draw each graph before you code.
- A reused input, a three-input graph, a logistic gradient, the adjoint of z, the passes of each mode.

```bash
# file: terminal
cd labs
uv run python problem_sets/check.py PS3
```

#### [week3#wrap-up] Three things to keep

*Week 3 · wrap-up*

- **Multiply, then add.** Multiply local derivatives along a path; add over the paths that reach the same node.
- **Reverse for a loss.** One backward pass gives the gradient for every weight; forward mode needs a pass per input.
- **p − y.** Sigmoid and cross-entropy together send back the prediction error to the score.

#### [week3#next] Week 4: Reverse mode in code

*Next*

Hand in Problem set 3. Keep your hand values from Lab 3.1: your engine's first test reproduces them.

## Week 4: Reverse mode in code

The chain rule with vectors and matrices, and a small automatic differentiation engine

#### [week4#title] Reverse mode in code

*SDAIA Academy · Calculus and Optimisation for Machine Learning*

Week 4: The chain rule with vectors and matrices, and a small automatic differentiation engine

Data & AI Foundations Track · 8 weeks + capstone · 1 session · 1 lab · FND-MTH-102

#### [week4#agenda] Reverse mode in code

*This week*

- **Session 1 · A reverse-mode engine**: Reverse mode multiplies vector-Jacobian products from the output back; a small engine records operations and replays them backwards

#### [week4#review-week3] Where does the second path go?

*Review of week 3*

**Question:** In h = (a · b) + a, at a = 4 and b = −1, what is ∂h/∂a?

- A. 1
- B. 0
- C. −1
- D. 5

#### [week4#lab-3-1-debrief] Four decimals by hand matched the code to about 1e-4

*Lab 3.1 · what your runs showed*

| Two-learner loss, reference run | Value |
| --- | --- |
| L | 0.7616 |
| dL/dw₁, dL/dw₂, dL/db | −0.0207, −0.0361, 0.1626 |
| Relative error against central differences | 1.4e-4 |
| Passes: forward mode, reverse mode | 3, 1 |

- **1.4e-4 is rounding.** Four decimals on a gradient of size 0.17 leave an error near 1e-4. A wrong formula shows up near 1e-1.
- **Most common slip.** Forgetting the ½ of the mean, which doubles every adjoint below it.

### Session 1: A reverse-mode engine

#### [week4#s1] A reverse-mode engine

*Session 1*

- Apply the chain rule to vector functions as products of Jacobians
- Design a scalar reverse-mode engine: nodes, local derivatives, topological order and accumulation
- Verify an engine with finite-difference checks and against a framework's gradients

#### [week4#jacobians-multiply] For vector functions, the chain rule multiplies Jacobians

*Block 1 · The chain rule with vectors*

- **Composition.** `J(F∘G)(x) = J_F(G(x)) · J_G(x)`

Shapes chain like the maps: `(m × k) · (k × n) = m × n`. Matrix products as composition of maps, from FND-MTH-101.

A network is a long composition: `L = ℓ(f₃(f₂(f₁(x))))`.

- **Two orders of multiplying.** **Forward**: start at the input, `J₃ (J₂ (J₁ v))`, one direction v at a time.
  **Reverse**: start at the loss, `((gᵀ J₃) J₂) J₁`, with gᵀ a row vector. Same product, very different cost.

#### [week4#vector-jacobian-products] A layer never builds its Jacobian; it passes a vector-Jacobian product

*Block 1 · What a backward pass passes*

| Operation | Upstream g = ∂L/∂y | Sends back |
| --- | --- | --- |
| Sum, `y = a + b` | g | g to a, g to b |
| Elementwise, `y = f(x)` | vector g | `g ⊙ f′(x)` |
| Dot, `y = w · x` | number g | `g x` to w, `g w` to x |
| Matrix–vector, `y = A x` | vector g | to x: Aᵀg; to A: g xᵀ |

> **Tensors.** Reverse mode on whole arrays, with broadcasting, is the subject of SPC-DLR-302. This course stays with scalars and small vectors.

#### [week4#check-gradient-shape] What shape is the weight gradient?

*Checkpoint 1*

**Question:** A layer computes y = A x with A of shape 3 × 5, then a scalar loss L follows. What shape is ∂L/∂A?

- A. 5
- B. 5 × 3
- C. 3 × 15
- D. 3 × 5

#### [week4#value-node] Each operation returns a node that knows how to pass its gradient back

*Block 2 · The engine's one idea*

```python
# file: a node the lab does not ask for
def sqrt(self):
    s = math.sqrt(self.data)
    out = Value(s, (self,), "sqrt")

    def _backward():
        # local derivative 1 / (2 sqrt(x))
        self.grad += out.grad / (2 * s)
    out._backward = _backward
    return out
```

- **Value**: the forward result, computed now.
- **Parents**: the inputs, so the graph can be walked.
- **_backward**: a closure that remembers what it needs (here `s`).
- **+=**: a node may receive from several users.

#### [week4#topological-order] backward() visits nodes in reverse topological order

*Block 2 · Walking the graph*

1. **1 · Order** list every node after its inputs
2. **2 · Seed** the output's grad = 1
3. **3 · Reverse** call each node's _backward, last to first

- **Why the order.** A node may pass its gradient on only after **every** user has sent its share. Reverse topological order guarantees it.
- **Why not recursion.** A recursive sort fails on a chain deeper than Python's recursion limit (about 1,000). The test suite has a 5,000-node chain: use a loop and a stack.

#### [week4#check-backward-order] What goes wrong in this backward pass?

*Checkpoint 2*

**Question:** An engine calls every node's _backward in the order the nodes were created (inputs first), after setting the output's grad to 1. What do the inputs end with?

- A. The correct gradients
- B. Zero, or only part of their gradient
- C. The correct gradients with flipped signs
- D. An error about the graph's order

#### [week4#two-referees] Two referees: finite differences and a framework's autograd

*Block 3 · Testing an engine*

```python
# file: reference with PyTorch autograd
import torch
f64 = dict(dtype=torch.float64, requires_grad=True)
x = torch.tensor(0.5, **f64)
y = torch.tensor(-1.2, **f64)
f = torch.tanh(x * y) + y ** 2
f.backward()
print(x.grad.item(), y.grad.item())
# -0.8539 -2.0442
```

| Referee | Agrees to about | Catches |
| --- | --- | --- |
| Central differences | 1e-9 | wrong formulas, any code |
| Framework autograd, float64 | 1e-13 | also errors of 1e-8 |

#### [week4#random-expressions] Ten tests, each aimed at one way an engine goes wrong

*Block 3 · What the suite tests*

- **Values and gradients** of +, ×, constants on either side
- **A reused node**, and one node used twice in a product
- **Each unary operation** against central differences
- **ReLU** on both sides of zero
- **20 random expressions**, seeded
- **Your Lab 3.1 loss**: the hand gradient again
- **A 5,000-node chain**, and **zero_grad**

- **What goes wrong at a kink.** ReLU has no derivative at 0. The engine must choose (0, here); a central difference straddling 0 returns 0.5. A "failed" check exactly at a kink is not a bug.

#### [week4#check-relu-kink] Is this engine wrong?

*Checkpoint 3*

**Question:** Every test passes except a finite-difference check of relu(x) at exactly x = 0, where the engine says 0 and the central difference says 0.5. What should you do?

- A. Change ReLU's derivative to 0.5
- B. Use a smaller step for the check
- C. Keep it, and check ReLU away from 0
- D. Remove ReLU from the engine

#### [week4#lab-4-1] A scalar reverse-mode engine

Lab 4.1 · 105 min

A scalar reverse-mode differentiation engine built and tested

[W4_1_autodiff_engine.ipynb](https://sdaiaacademy.github.io/fnd-mth-102/labs/W4_1_autodiff_engine.ipynb)

- [ ] Read the interface and run the test suite (5 min)
- [ ] Implement addition, multiplication and backward (25 min)
- [ ] Add power, exp, log, tanh and ReLU (20 min)
- [ ] Pass the finite-difference checks on random expressions (15 min)
- [ ] Compare your gradients with a framework's (15 min)
- [ ] Fix gradient accumulation for a reused node (15 min)
- [ ] Write three findings in the lab log (10 min)
- [ ] ✱ Stretch: Add a dot-product node with its vector-Jacobian product

#### [week4#problem-set-4] Problem set 4, and Test 1

*Before week 5*

- **PS4, 50 minutes**, AI-off first: three vector-Jacobian products, a composite Jacobian, a sin node for your engine.
- **Test 1, 90 minutes, supervised and AI-off**: 40 minutes on paper, 50 at the keyboard, weeks 1 to 4.
- To pass the course: at least 50% on each test.

```bash
# file: terminal
cd labs
uv run python problem_sets/check.py PS4
```

#### [week4#wrap-up] Three things to keep

*Week 4 · wrap-up*

- **Vector-Jacobian products.** Reverse mode multiplies from the loss end: a vector times each Jacobian, never a matrix product.
- **Node, closure, +=.** Each operation records its inputs and a closure that adds its share of the gradient.
- **Two referees.** Finite differences catch wrong formulas; a framework's autograd also catches small errors.

#### [week4#next] Week 5: Gradient descent

*Next*

Hand in Problem set 4 and sit Test 1. Week 5 uses the gradients: run the setup check again if it has been a while, since Lab 5.1 downloads public data.

## Week 5: Gradient descent

Step size, stochastic gradients, momentum and reading loss curves

#### [week5#title] Gradient descent

*SDAIA Academy · Calculus and Optimisation for Machine Learning*

Week 5: Step size, stochastic gradients, momentum and reading loss curves

Data & AI Foundations Track · 8 weeks + capstone · 1 session · 1 lab · FND-MTH-102

#### [week5#agenda] Gradient descent

*This week*

- **Session 1 · Descent and loss curves**: Gradient descent steps against the gradient; the step size, the batch and momentum decide whether a run converges, crawls or diverges

#### [week5#review-week4] What does a matrix–vector node send back to x?

*Review of week 4*

**Question:** A node computes y = A x with A of shape 4 × 6, and receives the upstream gradient g = ∂L/∂y. What does its backward pass add to x's gradient?

- A. A g
- B. g xᵀ
- C. The 4 × 6 Jacobian A
- D. Aᵀ g

#### [week5#lab-4-1-debrief] The engine agreed with the framework to about 1e-13

*Lab 4.1 · what your runs showed*

| Reference engine, solution run | Result |
| --- | --- |
| Test suite | 10 of 10 |
| Against central differences, 50 random expressions, worst | 3e-9 |
| Against PyTorch, Lab 3.1 loss | 6e-16 |
| Against PyTorch, ten random expressions, worst | 3e-13 |

- **One = instead of +=.** Overwriting in multiplication broke 6 of the 10 tests: every graph where a node reaches the output along more than one path.

### Session 1: Descent and loss curves

#### [week5#s1] Descent and loss curves

*Session 1*

- Apply batch, mini-batch stochastic and momentum gradient descent
- Describe what adaptive step methods, warm-up and decay change
- Diagnose divergence, oscillation and slow convergence from a loss curve

#### [week5#descent-step] Step against the gradient, by a rate you choose

*Block 1 · Gradient descent*

- **The update.** `w ← w − η ∇L(w)`

From week 2: −∇L is the steepest way down, and the local line promises a fall of about `η |∇L|²`.

The promise holds only while the step is small enough for the line to be right. The rate η decides that.

- **What goes wrong.** A rate that is too small wastes thousands of steps; one that is too large overshoots the valley and climbs out of it. Neither shows in the code: both show in the loss curve.

#### [week5#stable-rate] On a quadratic, each direction shrinks by |1 − η λ| per step

*Block 1 · How large is too large*

- For `L = ½ Σ λᵢ uᵢ²`, one step multiplies each `uᵢ` by `1 − η λᵢ`.
- **Stable** when every |1 − η λᵢ| < 1, that is **η < 2 / λ_max**.
- **Slow** when η λ_min is tiny: the flattest direction barely moves.
- Near the edge, 1 − η λ_max is close to −1: the steep direction flips sign every step.

Figure: 0.5 (u² + 10 v²), λ_max = 10, edge at η = 0.2; labs/_tools/make_figures.py.

#### [week5#check-stable-rate] Which rates will the analyst see diverge?

*Checkpoint 1*

**Question:** A least-squares loss has curvatures (eigenvalues of AᵀA / n) from 0.05 to 4. The analyst sweeps the rate over 0.01, 0.1, 0.4 and 0.6 with full-batch gradient descent. Which runs diverge?

- A. 0.1, 0.4 and 0.6
- B. 0.4 and 0.6
- C. 0.6 only
- D. None of them

#### [week5#stochastic-gradients] A mini-batch gradient is cheap, right on average, and noisy

*Block 2 · Stochastic and mini-batch gradients*

- The loss is a mean over n rows, so its gradient is a mean too.
- A random batch of b rows gives an **unbiased** estimate at **b / n** of the cost.
- Many cheap steps per pass over the data (an **epoch**), instead of one exact step.
- The noise does not vanish at the minimum: with a fixed rate, the weights keep moving.

- **Batch size as a dial.** b = 1 is stochastic gradient descent in its first form; b = n is full batch. Larger b: less noise per step, more cost per step. The lab uses b = 32.

#### [week5#momentum-path] Momentum keeps a velocity, so the flat direction speeds up

*Block 2 · Momentum*

- **Heavy ball.** `v ← β v + ∇L(w)`
  `w ← w − η v`

Steps that agree add up, to about `η / (1 − β)` in a steady slope; steps that flip sign cancel. With β = 0.9 it averages about the last ten gradients.

Figure: Same quadratic, same rate 0.04, 25 steps: loss 6.5 without momentum, 0.095 with β = 0.8.

#### [week5#adaptive-and-schedules] Adaptive methods scale each coordinate; schedules change the rate over time

*Block 2 · Adaptive steps and schedules*

- **Adaptive step (Adam, for example).** Keep running averages of g and g². Step each coordinate by `η · m̂ / (√v̂ + ε)`: about η per coordinate whatever the gradient's scale. Common defaults: β₁ = 0.9, β₂ = 0.999.
- **Warm-up, then decay.** **Warm-up**: start small for the first steps, when gradients are large and erratic. **Decay**: shrink the rate later, to narrow the noise band. Step, cosine and linear decays are all in use.

> **Awareness level.** Which adaptive optimiser is in common use changes; AdamW, scaling rules and newer methods are SPC-DLR-302's subject.

#### [week5#check-momentum] What does momentum change here?

*Checkpoint 2*

**Question:** Plain gradient descent at its largest stable rate zigzags across a narrow valley and crawls along it. The analyst adds momentum with β = 0.9 and keeps the rate. What should she expect first?

- A. Faster progress along the valley, possibly with overshoot; retune the rate
- B. Faster progress along the valley, and nothing else
- C. A lower final loss than the least-squares solution
- D. No change unless the batch size changes

#### [week5#reading-loss-curves] Each failure has a shape you can name from the curve

*Block 3 · Reading loss curves*

| The curve | Diagnosis | Likely cause | First fix |
| --- | --- | --- | --- |
| Grows, or turns into NaN | diverged | η above 2 / λ_max | divide η by 3 to 10 |
| Falls fast, then crawls far above the best | slow | η too small for λ_min; unscaled features | scale features; raise η; momentum |
| Falls, then a wide band that never narrows | noisy | small batch at a large constant η | decay the rate; larger batch |
| Saw-tooth on the way down | near the edge | η close to 2 / λ_max | lower η a little |
| Smooth fall to the reference | converged |  | keep it, and log it |

#### [week5#log-every-run] Log every run, not only the best one

*Block 3 · Reporting*

```
# file: out/experiments.csv (shape)
variant,learning_rate,seed,epochs_run,final_loss,diagnosis
batch,0.5,0,200,...,converged
batch,2.0,0,57,inf,diverged
```

- A sweep that reports only its best rate hides how fragile the method is.
- Every run gets its rate, its seed, its result and a diagnosis in words.
- The capstone is scored on this: every run, seed and failure, not the best result alone.

#### [week5#check-diagnose-curve] Name this run

*Checkpoint 3*

**Question:** On a dataset of household energy use, a run on raw, unscaled features falls from 900 to 520 in ten epochs, then sits near 500 for 190 more. The least-squares loss is 3. Any larger rate diverges. What is the diagnosis?

- A. Converged
- B. Slow: the rate fits the steepest direction and starves the flattest
- C. Noisy: use a larger batch
- D. Diverged: lower the rate

#### [week5#lab-5-1] Learning-rate sweeps

Lab 5.1 · 105 min

Learning-rate sweeps for three gradient descent variants on public regression data

[W5_1_lr_sweeps.ipynb](https://sdaiaacademy.github.io/fnd-mth-102/labs/W5_1_lr_sweeps.ipynb)

- [ ] Load the public data, split it and find the reference loss (15 min)
- [ ] Sweep the learning rate for batch gradient descent (20 min)
- [ ] Sweep it for mini-batch stochastic gradient descent (15 min)
- [ ] Sweep it for momentum (15 min)
- [ ] Run on unscaled features and diagnose (15 min)
- [ ] Write the experiment log with a diagnosis per run (15 min)
- [ ] Add warm-up and step decay to the noisy run (10 min)
- [ ] ✱ Stretch: Add an adaptive step method and compare its sweep

#### [week5#problem-set-5] Problem set 5: descent, a stable rate, momentum and four diagnoses

*Before week 6*

- **50 minutes**, graded, auto-checked: `labs/problem_sets/PS5.py`.
- **AI-off for the first attempt.**
- The checker runs your functions on its own small problems.

```bash
# file: terminal
cd labs
uv run python problem_sets/check.py PS5
```

#### [week5#wrap-up] Three things to keep

*Week 5 · wrap-up*

- **2 / λ_max.** The steepest curvature sets the largest stable rate; the flattest sets how slow it is.
- **Noise and velocity.** Mini-batches trade exactness for cheap steps; momentum averages steps; schedules shrink the noise band.
- **Curves are evidence.** Diagnose from the loss curve on a log scale, and log every run with its rate and seed.

#### [week5#next] Week 6: The shape of a loss

*Next*

Hand in Problem set 5. Keep your experiment log: week 6 explains its curvatures with the Hessian.

## Week 6: The shape of a loss

Convexity, saddle points, curvature and the largest stable step

#### [week6#title] The shape of a loss

*SDAIA Academy · Calculus and Optimisation for Machine Learning*

Week 6: Convexity, saddle points, curvature and the largest stable step

Data & AI Foundations Track · 8 weeks + capstone · 1 session · 1 lab · FND-MTH-102

#### [week6#agenda] The shape of a loss

*This week*

- **Session 1 · Convexity and curvature**: The Hessian tells a minimum from a saddle and bounds the step size; convex losses have no traps, non-convex ones do

#### [week6#review-week5] Which change fixes this run?

*Review of week 5*

**Question:** Mini-batch SGD at a constant rate reaches the reference loss quickly, then keeps jumping in a band 30% wide for the rest of training. Which single change narrows the band?

- A. Double the learning rate
- B. Train for twice as many epochs at the same rate
- C. Add more features
- D. Decay the rate after the loss reaches the band

#### [week6#lab-5-1-debrief] Scaling the features cut the curvature ratio from 2.7e11 to 12

*Lab 5.1 · what your runs showed*

| Airfoil data, reference run | Diagnosis |
| --- | --- |
| Batch, rate 0.01: loss 152 after 200 epochs | slow |
| Batch, rate 1.0, above the edge 0.948 | diverged |
| SGD, rate 0.3: last 20 epochs spread 2.95 | noisy |
| Momentum, rate 0.001: loss 125 | slow |
| Unscaled, rate 1e-7: loss 4,193; 1e-6 diverged at epoch 123 | slow, diverged |

- **Reference loss 11.31.** Least squares on 1,200 training rows; the yardstick every run was judged against.
- **A schedule helped.** Warm-up and step decay cut the noisy run's spread from 2.95 to 0.014.

### Session 1: Convexity and curvature

#### [week6#s1] Convexity and curvature

*Session 1*

- Distinguish convex from non-convex objectives with a test you can run
- Classify critical points from the eigenvalues of the Hessian
- Relate the largest stable step to the largest curvature

#### [week6#convex-definition] A convex function lies below every chord, so it has no traps

*Block 1 · Convexity*

- **Definition.** `f(t a + (1 − t) b) ≤ t f(a) + (1 − t) f(b)` for all a, b and t in [0, 1]

Twice differentiable: convex exactly when `f″ ≥ 0` everywhere, or in several variables when the Hessian has no negative eigenvalue anywhere.

- **Why it matters.** On a convex loss every local minimum is a global minimum, and a small enough step of gradient descent reaches it. Least squares and logistic regression are convex in their weights. A network's loss is not.

#### [week6#minima-and-saddles] Non-convex losses have local minima, saddles and plateaus

*Block 1 · What non-convex looks like*

- **Local minimum.** Lower than its neighbours, higher than the best point. Descent stops there.
- **Saddle point.** Gradient zero, curving up one way and down another. In many dimensions, saddles far outnumber minima.
- **Plateau.** A wide region where the gradient is close to zero. Steps become tiny; the run looks converged and is not.

> **What goes wrong.** Reading "the loss stopped falling" as "we found the minimum". A saddle or a plateau stops the loss too.

#### [week6#check-convex] What does convexity guarantee here?

*Checkpoint 1*

**Question:** The charity's analyst fits a logistic regression twice, from two different starting weights, with a small enough step. Both runs stop. What can she conclude about the two end points?

- A. They may be different local minima
- B. Both runs reach the same lowest loss
- C. Any step size gives the same end point
- D. One of them may be a saddle point

#### [week6#hessian-curvatures] The Hessian's eigenvalues are the curvatures along its eigenvectors

*Block 2 · Curvature*

The Hessian **H** holds the second partial derivatives: `Hᵢⱼ = ∂²f / ∂wᵢ∂wⱼ`. It is symmetric.

- **Second-order Taylor.** `f(w + d) ≈ f(w) + ∇f · d + ½ dᵀ H d`

Along an eigenvector with eigenvalue λ, the function curves like `½ λ t²`.

| At a critical point, eigenvalues | Kind |
| --- | --- |
| all positive | minimum |
| all negative | maximum |
| both signs | saddle |
| some zero | the test cannot tell |

#### [week6#saddle-escape] Exactly on a saddle's ridge, descent stops; a nudge lets it escape

*Block 2 · Saddles in practice*

Figure: f = ½x² − ½y² + 0.1y⁴, step 0.2, 120 steps; labs/_tools/make_figures.py.

On the axis the y-component of the gradient is exactly 0: descent walks into the saddle at (0, 0).

From y = 0.01 it ends at (0, 1.581), a minimum (√2.5).

> **Noise helps.** Mini-batch gradients rarely sit exactly on a ridge.

#### [week6#check-hessian-kind] What kind of point is this?

*Checkpoint 2*

**Question:** At a point where the gradient of a two-weight loss is zero, the Hessian is [[3, 0], [0, −0.2]]. The run has stopped there. What should the analyst say?

- A. It is a minimum, since 3 is large and positive
- B. The test cannot tell: −0.2 is too small to count as negative
- C. It is a saddle: lower loss lies along the second weight
- D. It is a maximum

#### [week6#largest-stable-step] The largest curvature bounds the step; the ratio of curvatures sets the pace

*Block 3 · Step size and curvature*

- Near a minimum the loss is about quadratic with Hessian H: week 5's rule holds with λ_max of H. **Step < 2 / λ_max.**
- At the best fixed step, the flattest direction shrinks by about `1 − 2/κ` per step, with **κ = λ_max / λ_min**, the condition number.
- Steps needed grow roughly in proportion to κ.

| Lab 5.1, airfoil data | κ |
| --- | --- |
| Raw features | 2.7 × 10¹¹ |
| Standardised features | 12 |

#### [week6#newton-step] A Newton step divides by the curvature instead of guessing a rate

*Block 3 · Using the curvature*

- **Newton.** Solve `H d = −∇f`, then `w ← w + d`. On a quadratic it lands on the minimum in one step, whatever κ is.
- **Why not always.** H has n² entries: for a million weights, a trillion. At a saddle a Newton step goes towards the saddle, not away. Training uses first-order methods; Newton is the yardstick.

#### [week6#check-condition] What will the sweep show?

*Checkpoint 3*

**Question:** A convex quadratic loss has curvatures 0.5 and 50. The analyst runs plain gradient descent at steps 0.01, 0.039 and 0.041. What does she see?

- A. 0.041 diverges; the other two converge, slowly along the flat direction
- B. All three converge quickly
- C. All three converge, since a convex loss cannot diverge whatever the step
- D. 0.039 and 0.041 diverge

#### [week6#lab-6-1] Six loss surfaces

Lab 6.1 · 105 min

Six loss surfaces with gradient-descent traces, classified and annotated

[W6_1_loss_surfaces.ipynb](https://sdaiaacademy.github.io/fnd-mth-102/labs/W6_1_loss_surfaces.ipynb)

- [ ] Plot the six surfaces as contour maps (10 min)
- [ ] Classify critical points from the Hessian (20 min)
- [ ] Test each surface for convexity (15 min)
- [ ] Trace gradient descent below, at and above the stable step (20 min)
- [ ] Start on and off the saddle's ridge (10 min)
- [ ] Annotate each plot against the reference (20 min)
- [ ] Write three findings in the lab log (10 min)
- [ ] ✱ Stretch: Take Newton steps on the valley and compare with gradient descent

#### [week6#problem-set-6] Problem set 6: a Hessian, two critical points, an edge and three verdicts

*Before week 7*

- **50 minutes**, graded, auto-checked: `labs/problem_sets/PS6.py`.
- **AI-off for the first attempt.**
- Convexity verdicts are checked on the whole plane: think before you sample.

```bash
# file: terminal
cd labs
uv run python problem_sets/check.py PS6
```

#### [week6#wrap-up] Three things to keep

*Week 6 · wrap-up*

- **Convex: no traps.** Every local minimum is global. Least squares and logistic regression are convex; networks are not.
- **Hessian signs.** All positive: minimum. Mixed: saddle. Stopping is not proof of a minimum.
- **2 / λ_max and κ.** The steepest curvature bounds the step; the condition number sets how many steps.

#### [week6#next] Week 7: Stable losses and penalties

*Next*

Hand in Problem set 6. Week 7 makes the losses you will train in the capstone safe to compute.

## Week 7: Stable losses and penalties

Log-sum-exp, softmax cross-entropy and regularisation as a penalty

#### [week7#title] Stable losses and penalties

*SDAIA Academy · Calculus and Optimisation for Machine Learning*

Week 7: Log-sum-exp, softmax cross-entropy and regularisation as a penalty

Data & AI Foundations Track · 8 weeks + capstone · 1 session · 1 lab · FND-MTH-102

#### [week7#agenda] Stable losses and penalties

*This week*

- **Session 1 · Stable losses and penalties**: Shift logits before exponentiating, use the clean gradient of softmax cross-entropy, and add a penalty to change the shape of the loss

#### [week7#review-week6] Why did this run stop?

*Review of week 6*

**Question:** Gradient descent on a two-weight loss stops moving at a point where the gradient is zero. The Hessian there has eigenvalues 6 and −1. What should the report say?

- A. The run found a minimum and can stop
- B. The run is stuck at a saddle; a small push lowers the loss
- C. The step was too large and must be halved
- D. The loss is non-convex, so nothing better can be found here at all

#### [week7#lab-6-1-debrief] Two of six surfaces were convex, and one of those still diverged

*Lab 6.1 · what your annotations showed*

| Surface | Convex | Trace ended |
| --- | --- | --- |
| bowl | yes | global minimum |
| narrow, step 0.085 | yes | diverged |
| valley | no | global minimum |
| saddle, from (1, 0) | no | stuck at the saddle |
| bumpy | no | a local minimum |
| plateau | no | did not move |

- **The edge on narrow: 2 / 25 = 0.08.** Loss after 60 steps: 0.0096 at 0.05, 0.60 at 0.079, 242 at 0.081.
- **The ridge.** From (1, 0) descent stopped at (0, 0); from (1, 0.001) it reached (0, 1.414).

### Session 1: Stable losses and penalties

#### [week7#s1] Stable losses and penalties

*Session 1*

- Compute softmax and log-sum-exp without overflow
- Derive and check the gradient of softmax cross-entropy
- Explain regularisation as a penalty that reshapes the loss and bounds the weights

#### [week7#float-range] Shift the logits by their maximum before exponentiating

*Block 1 · Overflow*

A 64-bit float stops at about 1.8 × 10³⁰⁸, so `exp(z)` is `inf` for z above about 709.8. A model's logits can reach that.

- **Softmax ignores a common shift.** `softmax(z) = softmax(z − m)` for any number m. Take `m = max z`: the largest exponent is then `exp(0) = 1`.

| Logits (3, 1, −1) | value |
| --- | --- |
| exp of the shifted (0, −2, −4) | 1, 0.135, 0.018 |
| softmax | 0.867, 0.117, 0.016 |
| log-sum-exp = 3 + log(1 + e⁻² + e⁻⁴) | 3.143 |

#### [week7#ce-from-logits] Compute cross-entropy from logits, never as the log of a probability

*Block 1 · The loss*

- **Stable form.** `CE(z, y) = logsumexp(z) − z_y` No probability is formed, so no probability can round to 0.
- **Fragile form.** `−log(softmax(z)_y)` If the probability of the label rounds to 0, the loss is `inf` and every gradient after it is `nan`.

> **In practice.** Libraries offer a loss that takes logits for exactly this reason; pass it logits, not probabilities.

#### [week7#check-overflow] Which computation is safe?

*Checkpoint 1*

**Question:** A model gives the logits (900, 905) for a learner with label 0. Which way of computing the cross-entropy returns a finite, correct number in 64-bit floats?

- A. Take −log of exp(900) / (exp(900) + exp(905))
- B. Clip both logits to 709, then use the naive formula
- C. Compute 905 + log(e⁻⁵ + 1) − 900
- D. Divide both logits by 905, then use the naive formula

#### [week7#softmax-jacobian] The gradient of softmax cross-entropy is probabilities minus the label

*Block 2 · The gradient*

The Jacobian of softmax is `∂pᵢ/∂zⱼ = pᵢ (δᵢⱼ − pⱼ)`: `diag(p) − p pᵀ`.

Differentiate `logsumexp(z) − z_y` directly: the first term gives p, the second gives the one-hot vector `e_y`.

- **Result.** `∂CE / ∂z = p − e_y`

| Logits (2, 0, 1), label 2 | value |
| --- | --- |
| p | 0.665, 0.090, 0.245 |
| e_y | 0, 0, 1 |
| gradient p − e_y | 0.665, 0.090, −0.755 |

#### [week7#check-gradient] What is the gradient?

*Checkpoint 2*

**Question:** For one learner the model's probabilities are (0.1, 0.7, 0.2) and the true level is 0. What is the gradient of the cross-entropy with respect to the three logits?

- A. (0.1, −0.3, 0.2)
- B. (0.9, −0.7, −0.2)
- C. (−10, 0, 0)
- D. (−0.9, 0.7, 0.2)

#### [week7#penalty-shape] On separable data the plain loss falls for ever; a penalty gives it a minimum

*Block 3 · Penalties*

Figure: Four points x = −2, −1, 1, 2 with labels 0, 0, 1, 1; labs/_tools/make_figures.py.

Any c > 0 classifies all four points; a larger c is more confident, and the loss is lower. No finite c is best.

- **Penalised loss.** `L(w) + (λ/2) ‖w‖²`

Minimum at c = 2.43 for λ = 0.02, and at c = 1.46 for λ = 0.1.

#### [week7#penalty-curvature] The penalty adds λ to every curvature, so the minimum becomes unique

*Block 3 · What the penalty changes*

- **Hessian.** `H + λI`: every eigenvalue rises by λ. A convex loss becomes strictly convex, with one minimiser.
- **Gradient.** `∇L + λw`: each step also shrinks the weights by the factor `1 − ηλ`, which is why it is called weight decay.
- **Price.** The minimiser is no longer the plain loss's best point: the training loss rises. Whether held-out loss falls is a measurement, not a promise.

#### [week7#check-penalty] What does the penalty do to this Hessian?

*Checkpoint 3*

**Question:** At its minimum, a plain loss in two weights has Hessian eigenvalues 0 and 4. The analyst adds the penalty (0.5/2)‖w‖². What are the new eigenvalues, and what changes?

- A. 0.5 and 4.5: the minimiser is now unique
- B. 0 and 2: the steep direction is flattened
- C. 0.5 and 4: only the flat direction changes
- D. 0 and 4: a penalty only moves the minimum

#### [week7#lab-7-1] Stable softmax and a penalty

Lab 7.1 · 105 min

A stable three-level placement model, trained with and without an L2 penalty

[W7_1_stable_softmax.ipynb](https://sdaiaacademy.github.io/fnd-mth-102/labs/W7_1_stable_softmax.ipynb)

- [ ] Watch naive softmax overflow on the planted logits (10 min)
- [ ] Write stable log-sum-exp, softmax and cross-entropy (20 min)
- [ ] Check the gradient of softmax cross-entropy (15 min)
- [ ] Train the placement model with gradient descent (20 min)
- [ ] Add an L2 penalty and sweep its strength (15 min)
- [ ] Compare weight norms and held-out loss (15 min)
- [ ] Write three findings in the lab log (10 min)
- [ ] ✱ Stretch: Show that the penalised loss is strictly convex while the plain one is not

#### [week7#problem-set-7] Problem set 7: stable losses, two gradients and a penalised minimum

*Before week 8*

- **50 minutes**, graded, auto-checked: `labs/problem_sets/PS7.py`.
- **AI-off for the first attempt.**
- The checker feeds logits of size 1000: a naive formula fails it.

```bash
# file: terminal
cd labs
uv run python problem_sets/check.py PS7
```

#### [week7#wrap-up] Three things to keep

*Week 7 · wrap-up*

- **Shift, then exp.** Log-sum-exp with the row's maximum; cross-entropy from logits.
- **p − e_y.** The gradient of softmax cross-entropy with respect to the logits.
- **A penalty is a choice.** It bounds the weights and makes the minimum unique; its effect on held-out loss is measured.

#### [week7#next] Week 8: Constraints

*Next*

Hand in Problem set 7. Week 8 replaces a soft penalty with a hard limit: a fixed budget, and the price of each extra thousand.

## Week 8: Constraints

Lagrange multipliers, projected gradients and the price of a constraint

#### [week8#title] Constraints

*SDAIA Academy · Calculus and Optimisation for Machine Learning*

Week 8: Lagrange multipliers, projected gradients and the price of a constraint

Data & AI Foundations Track · 8 weeks + capstone · 1 session · 1 lab · FND-MTH-102

#### [week8#agenda] Constraints

*This week*

- **Session 1 · Constraints and prices**: A constraint moves the optimum; the Lagrange multiplier prices it, and projected gradients reach the same answer by stepping and projecting

#### [week8#review-week7] Where did this loss go wrong?

*Review of week 7*

**Question:** A training run prints a finite loss for 300 steps, then `inf`, then `nan` for ever. The code computes `-log(softmax(z)[y])`. What is the most likely cause?

- A. The learning rate is too low
- B. The L2 penalty is too strong
- C. A label's probability rounded to 0, and its log is −inf
- D. The batches were not shuffled between the epochs of training

#### [week8#lab-7-1-debrief] The penalty bounded the weights; whether it lowered held-out loss depended on the labels

*Lab 7.1 · what your runs showed*

| λ | ‖W‖ | held-out loss, first results | held-out loss, re-test |
| --- | --- | --- | --- |
| 0 | 16.22 | 0.071 | 0.700 |
| 0.001 | 11.10 | 0.113 | 0.541 |
| 0.01 | 4.90 | 0.273 | 0.462 |
| 0.1 | 1.70 | 0.586 | 0.653 |
| 1 | 0.39 | 0.927 | 0.943 |

- **Naive: nan, nan, inf.** Stable cross-entropy on the planted rows: 22.00. Gradient check: 9e-10.
- **Plain norm still growing.** 3.70 at step 50, 16.22 at step 2,000. Smallest Hessian eigenvalue: about 0 plain, 0.01 with λ = 0.01.

### Session 1: Constraints and prices

#### [week8#s1] Constraints and prices

*Session 1*

- Solve a small equality-constrained problem with the Lagrangian
- Read a multiplier as the price of its constraint
- Explain how optimising a convenient proxy can harm the people a model serves

#### [week8#lagrangian] At a constrained optimum, the gradient points across the constraint

*Block 1 · The Lagrangian*

Maximise `f(x)` subject to `g(x) = c`. Moving along the constraint cannot improve f, so `∇f` is parallel to `∇g`:

- **Lagrange conditions.** `∇f = λ ∇g` and `g(x) = c` Equivalently: set the derivatives of `f − λ (g − c)` to zero.

| Maximise 2 log x + log y, x + y = 6 | value |
| --- | --- |
| conditions: 2 / x = λ, 1 / y = λ | x = 2y |
| constraint: 3y = 6 | y = 2, x = 4 |
| multiplier | λ = 0.5 |

#### [week8#multiplier-price] The multiplier is the price of the constraint: value per extra unit

*Block 1 · The multiplier*

- **Reading.** If the best value is `V(c)`, then `dV/dc = λ`. One more unit of budget is worth about λ units of the objective, for small changes.
- **In the example.** λ = 0.5: raising the total from 6 to 6.1 adds about 0.05 to the best value. Its units are the objective's units per unit of the constraint.

> **Why it matters.** A finance officer asks "what would another thousand buy?" The multiplier answers without re-solving.

#### [week8#check-lagrange] What does the multiplier say?

*Checkpoint 1*

**Question:** A training team's best accuracy under a compute limit of 100 hours has multiplier λ = 0.002 accuracy points per hour. The limit is cut to 90 hours. What should the team expect?

- A. Accuracy falls by 0.002
- B. Accuracy falls by about 0.02
- C. Accuracy is unchanged and λ rises
- D. Accuracy falls by 0.0002

#### [week8#equal-margins] With budgets that cannot be negative, a programme can get nothing

*Block 2 · Limits that bind*

Figure: Three programmes, value k log(1 + x/s), budget 30; labs/_tools/make_figures.py.

Each funded programme gets `x = k/λ − s`: where its margin meets λ.

C's margin at zero is 0.8, below λ = 1.163: its first unit is worth less than the last unit anywhere else, so `x_C = 0`.

> **Find λ** by bisection: raise it until the budgets fit.

#### [week8#projected-gradient] Projected gradient: step uphill, then return to the nearest allowed point

*Block 2 · A second way*

1. `v = x + η ∇f(x)`: an ordinary gradient step.
2. `x = project(v)`: the nearest point with `x ≥ 0` and sum = budget, which is `max(v − θ, 0)` for one θ.
3. Repeat until x stops moving.

- **Week 5 applies.** Too small a step crawls; too large a step overshoots and never settles. Both solutions should agree; if they do not, one of them is wrong.

#### [week8#check-corner] Who gets funded?

*Checkpoint 2*

**Question:** At the optimum of an allocation, λ = 1.5. A fourth programme has margin k / s = 1.2 at zero budget. What should it receive?

- A. An equal share of the budget
- B. Enough budget to raise its margin up to λ = 1.5 first
- C. 1.5 units of budget
- D. Nothing: its first unit is worth less than λ

#### [week8#proxy-harm] Optimising a convenient proxy can harm the people a model serves

*Block 3 · Responsible use*

- **The proxy.** One number that is easy to count and maximise. The optimiser does exactly what it is told, including leaving out whoever the number does not see.
- **The fix is a decision.** A floor, or weights by need, changes the answer. Which one, and how much, is policy: the mathematics prices it, it does not choose it.
- **Separate the roles.** The analyst proposes options with their prices; the finance officer approves the budget; the programme lead decides the weights.

#### [week8#check-roles] Who decides?

*Checkpoint 3*

**Question:** An analyst finds that adding a floor for two programmes costs 0.6 units of the objective per unit of budget. Which next step keeps the duties separate?

- A. Present both options with their prices to the person who sets the weights
- B. Adopt the floor, since it is fairer
- C. Drop the floor, since it lowers the objective
- D. Approve the cheaper option now and tell the finance officer afterwards

#### [week8#lab-8-1] A budget allocation two ways

Lab 8.1 · 105 min

A budget allocation solved with Lagrange multipliers and with projected gradients

[W8_1_budget_allocation.ipynb](https://sdaiaacademy.github.io/fnd-mth-102/labs/W8_1_budget_allocation.ipynb)

- [ ] Write the Lagrangian for two districts and solve it by hand (15 min)
- [ ] Solve the five-district problem from the Lagrange conditions (20 min)
- [ ] Solve it again with projected gradient ascent (20 min)
- [ ] Compare the answers and read the multiplier (15 min)
- [ ] Find who the proxy leaves out and add a floor (20 min)
- [ ] Write the recommendation for the finance officer (15 min)
- [ ] ✱ Stretch: Weight each district by need and see how the multiplier moves

#### [week8#problem-set-8] Problem set 8: three constrained problems and their prices

*Before Test 2*

- **50 minutes**, graded, auto-checked: `labs/problem_sets/PS8.py`.
- **AI-off for the first attempt.**
- Read every multiplier as a price, in words, as in the lab.

```bash
# file: terminal
cd labs
uv run python problem_sets/check.py PS8
```

#### [week8#wrap-up] Three things to keep

*Week 8 · wrap-up*

- **Equal margins.** At the optimum every funded option has the same marginal value, λ.
- **λ is a price.** The best value moves by about λ per unit of the constraint.
- **Price, then decide.** The analyst prices the options; the people accountable choose.

#### [week8#next] Test 2, then the capstone

*Next*

Hand in Problem set 8. The capstone brief is in `labs/capstone/BRIEF.md`: a two-layer network trained with your own engine, every run reported.

## Capstone: Capstone oral checks

A two-layer network trained with your own engine, defended in ten minutes with the questions inside them

#### [capstone#title] Capstone oral checks

*SDAIA Academy · Calculus and Optimisation for Machine Learning*

Capstone: A two-layer network trained with your own engine, defended in ten minutes with the questions inside them

Data & AI Foundations Track · 8 weeks + capstone · 10 min talk and demo · FND-MTH-102

#### [capstone#agenda] How the day runs

*Capstone*

- **Brief and planning clinic**: the rubric, the oral check, each participant's plan
- **Build, train and write**: engine, network, runs, log and report
- **Gradient check and a test oral**: in pairs, before the real one
- **Oral checks**: one at a time, questions included
- **Written feedback**: the trainer's review of each submission

#### [capstone#rubric] The rubric

*Capstone score · 30% of the grade*

| Item | Value |
| --- | --- |
| Engine tests and gradient checks pass | 25% |
| Public data, held-out split, reproducible | 20% |
| Three optimisers, 3+ seeds, with curves | 25% |
| Every run, seed and failure reported | 15% |
| Each failed run diagnosed from its log | 15% |

> **Report every run.** A diagnosed failure earns marks; a hidden one costs them.

#### [capstone#format] 10 minutes for each oral check, questions included

*The oral check · 10% of the course grade*

- **1 · A backward pass.** The trainer writes a small expression; you trace it through your own engine.
- **2 · A failed run.** Open your log, pick one, explain its curve and what you changed.
- **3 · One choice.** One line of your code, and why it is written that way.

> **Practise in pairs.** In the 30-minute block before the orals, ask each other a backward pass on an expression you did not prepare.

#### [capstone#running-order] Who has the oral check when

*Running order*

| # | Participant | Repository |
| --- | --- | --- |
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |
| 4 |  |  |
| 5 |  |  |
| 6 |  |  |
| 7 |  |  |
| 8 |  |  |

#### [capstone#final-prep] Gradient check and a test oral · 30 minutes

*Before the first oral check*

Run the engine tests and the gradient check once more, push the repository, then practise in pairs.

#### [capstone#now-presenting] Oral check, questions included

*Oral check*

#### [capstone#review] What the submissions showed

*Trainer review*

- **Strongest patterns.** What several submissions got right.
- **Common gaps.** What most would fix next: often the diagnosis of failed runs.
- **One surprise.** Something nobody expected in the logs.

#### [capstone#what-next] Keep going

*After the course*

- **Your engine stays yours:** keep the repository reproducible, and add the next experiment your log suggests
- **Next Academy course:** SPC-DLR-302 Deep Learning from First Principles, where differentiation moves from scalars to tensors and modern optimisers take over

#### [capstone#thanks] Thank you

*SDAIA Academy*

Calculus and Optimisation for Machine Learning
