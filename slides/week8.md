---
course: "Calculus and Optimisation for Machine Learning"
code: "FND-MTH-102"
version: "2027.1.0"
deck: "week8"
day: 8
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/fnd-mth-102/slides/week8.html"
---

# Calculus and Optimisation for Machine Learning

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
