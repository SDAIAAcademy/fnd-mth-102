---
course: "Calculus and Optimisation for Machine Learning"
code: "FND-MTH-102"
version: "2027.1.0"
deck: "week4"
day: 4
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/fnd-mth-102/slides/week4.html"
---

# Calculus and Optimisation for Machine Learning

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
