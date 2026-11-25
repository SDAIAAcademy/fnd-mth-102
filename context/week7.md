---
course: "Calculus and Optimisation for Machine Learning"
code: "FND-MTH-102"
version: "2027.1.0"
deck: "week7"
day: 7
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/fnd-mth-102/slides/week7.html"
---

# Calculus and Optimisation for Machine Learning

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
