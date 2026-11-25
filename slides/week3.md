---
course: "Calculus and Optimisation for Machine Learning"
code: "FND-MTH-102"
version: "2027.1.0"
deck: "week3"
day: 3
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/fnd-mth-102/slides/week3.html"
---

# Calculus and Optimisation for Machine Learning

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
