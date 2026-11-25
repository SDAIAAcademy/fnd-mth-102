---
course: "Calculus and Optimisation for Machine Learning"
code: "FND-MTH-102"
version: "2027.1.0"
deck: "week2"
day: 2
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/fnd-mth-102/slides/week2.html"
---

# Calculus and Optimisation for Machine Learning

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
