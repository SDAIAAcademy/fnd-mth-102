---
course: "Calculus and Optimisation for Machine Learning"
code: "FND-MTH-102"
version: "2027.1.0"
deck: "week1"
day: 1
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/fnd-mth-102/slides/week1.html"
---

# Calculus and Optimisation for Machine Learning

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
