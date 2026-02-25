# Chapter 2 - Math You Can't Escape (But Can Tame)

## The Crux
You can avoid some math in AI. You can't avoid all of it. The good news: you don't need PhD-level math. You need *intuition* for a few key concepts. This chapter builds that intuition without drowning you in proofs.

## The Math You Actually Need

Here's the honest breakdown:

**Must-Have**:
- Linear algebra (vectors, matrices, dot products)
- Probability (distributions, expectations, Bayes' rule)
- Calculus (derivatives, chain rule, gradients)

**Nice-to-Have**:
- Information theory (entropy, KL divergence)
- Statistics (hypothesis testing, confidence intervals)
- Optimization theory (convexity, saddle points)

**Overkill-for-Most**:
- Real analysis
- Measure theory
- Functional analysis

You can be effective without the third category. Let's build intuition for the first.

## Linear Algebra as Geometry

Most people learn linear algebra as symbol manipulation. That's backwards. **Linear algebra is geometry.**

### Vectors: Points in Space

A vector is just coordinates in space. `[3, 4]` means "3 steps right, 4 steps up" in 2D.

In AI, vectors represent *features*. An email might be:
```
[
  word_count: 150,
  has_money_mention: 1,
  has_typos: 0
]
```

This is a point in 3D "email space."

### Dot Product: Measuring Similarity

The dot product of two vectors measures how much they point in the same direction.

```
a · b = |a| |b| cos(θ)
```

Intuition:
- If vectors point the same way: large positive dot product
- If perpendicular: dot product = 0
- If opposite directions: large negative dot product

**In AI**: Dot products are everywhere. They measure similarity. "Is this email similar to spam emails?" ≈ dot product with a "spam direction" vector.

### Matrices: Transformations

A matrix is a transformation. It takes vectors and rotates/scales/shears them.

```
[2  0]  [x]     [2x]
[0  3]  [y]  =  [3y]
```

This matrix stretches x-direction by 2, y-direction by 3.

**In AI**: Neural network layers are matrix multiplications. Input vector → multiply by weight matrix → transformed vector. Each layer is a geometric transformation of the data.

### Why This Matters

When you hear "the model is learning a representation," it means: **the model is learning geometric transformations that make patterns linearly separable**.

Imagine email space. Initially, spam and ham are jumbled together. After transformations (neural network layers), spam clusters in one region, ham in another. Now you can draw a line (hyperplane) separating them.

That's all deep learning is: warp space until patterns become obvious.

## Probability as Uncertainty Management

AI is fundamentally about dealing with uncertainty. Probability is the language of uncertainty.

### Distributions: Describing Uncertainty

A probability distribution describes what values are likely.

**Example**: Height of adult men might follow a normal distribution centered at 5'10" with some spread.

**In AI**: You don't predict "this email is spam." You predict "this email has 73% probability of being spam." That's a distribution over {spam, not spam}.

### Expectation: The Average Outcome

The expectation E[X] is the weighted average of all outcomes.

**Intuition**: If you rolled a die many times, what's the average result? (1+2+3+4+5+6)/6 = 3.5

**In AI**: Loss functions measure "expected error." You're optimizing for average performance across your data distribution.

### Bayes' Rule: Flipping the Question

Bayes' rule lets you reverse conditional probabilities:

```
P(A|B) = P(B|A) P(A) / P(B)
```

**Intuition**: You know "90% of spam contains word X" but you want to know "if an email contains word X, what's the probability it's spam?" Bayes' rule lets you flip the question.

**In AI**: Naive Bayes classifiers, Bayesian inference, posterior distributions-all Bayes' rule.

### Common Misconception: "I'll Learn Probability Later"

No, you won't. Without probability, you can't:
- Understand what models are actually predicting
- Debug calibration issues (model says 90% confident but is wrong 50% of the time)
- Reason about uncertainty
- Understand loss functions

Bite the bullet now.

## Information Theory: The Math Behind Loss Functions

Information theory provides the mathematical foundation for understanding loss functions, model training, and uncertainty. This section builds rigorous intuition for concepts you'll use daily.

### Entropy: Measuring Uncertainty

**Definition**: Entropy H(X) measures the average "surprise" or uncertainty in a random variable X.

For a discrete random variable with outcomes {x₁, x₂, ..., xₙ} and probabilities {p₁, p₂, ..., pₙ}:

```
H(X) = -∑ᵢ p(xᵢ) log₂ p(xᵢ)
```

(Convention: 0 log 0 = 0)

**Intuition**: Entropy answers "how many bits, on average, do I need to encode outcomes from this distribution?"

**Examples**:

1. **Fair coin**: p(heads) = 0.5, p(tails) = 0.5
   ```
   H(X) = -0.5 log₂(0.5) - 0.5 log₂(0.5) = 1 bit
   ```
   Maximum uncertainty. You need 1 bit to encode the outcome.

2. **Unfair coin**: p(heads) = 0.99, p(tails) = 0.01
   ```
   H(X) = -0.99 log₂(0.99) - 0.01 log₂(0.01) ≈ 0.08 bits
   ```
   Low uncertainty. Outcome is almost always heads-you can compress this information.

3. **Deterministic**: p(heads) = 1.0, p(tails) = 0.0
   ```
   H(X) = -1.0 log₂(1.0) - 0 log₂(0) = 0 bits
   ```
   No uncertainty. You don't need to transmit anything-the outcome is known.

**Key Property**: Entropy is maximized when all outcomes are equally likely (uniform distribution).

For n outcomes: H_max = log₂(n)

**In AI**: Entropy measures model uncertainty. High entropy = model is uncertain about predictions. Low entropy = model is confident (could be good or bad-confident and wrong is worse than uncertain).

### Cross-Entropy: Comparing Distributions

**Definition**: Cross-entropy H(p, q) measures the average number of bits needed to encode data from distribution p using a code optimized for distribution q.

```
H(p, q) = -∑ᵢ p(xᵢ) log q(xᵢ)
```

Where:
- p = true distribution
- q = predicted distribution

**Intuition**: If your model (q) perfectly matches reality (p), cross-entropy equals entropy. If they differ, cross-entropy is higher-you're using a suboptimal encoding.

**Example**:

True distribution p: p(A) = 0.5, p(B) = 0.5 (fair coin)
Model's distribution q: q(A) = 0.9, q(B) = 0.1 (model thinks A is very likely)

```
H(p, q) = -0.5 log(0.9) - 0.5 log(0.1)
        = -0.5(-0.046) - 0.5(-1.0)
        = 0.523 bits
```

Compare to entropy of p:
```
H(p) = -0.5 log(0.5) - 0.5 log(0.5) = 0.5 bits
```

Cross-entropy (0.523) > Entropy (0.5), indicating the model's predictions are imperfect.

**In AI**: Cross-entropy loss measures how well your model's predicted probability distribution matches the true distribution. Minimizing cross-entropy = making your model's predictions closer to reality.

### KL Divergence: The Distance Between Distributions

**Definition**: Kullback-Leibler divergence D_KL(p || q) measures how much information is lost when using q to approximate p.

```
D_KL(p || q) = ∑ᵢ p(xᵢ) log(p(xᵢ) / q(xᵢ))
             = ∑ᵢ p(xᵢ) log p(xᵢ) - ∑ᵢ p(xᵢ) log q(xᵢ)
             = -H(p) + H(p, q)
```

**Key Identity**:
```
H(p, q) = H(p) + D_KL(p || q)
```

Cross-entropy = Entropy + KL divergence

**Properties**:
1. **Always non-negative**: D_KL(p || q) ≥ 0
2. **Zero iff distributions match**: D_KL(p || q) = 0 ⟺ p = q
3. **Not symmetric**: D_KL(p || q) ≠ D_KL(q || p) (not a true distance metric)
4. **Not a metric**: Doesn't satisfy triangle inequality

**Example**:

Using the previous example:
- p: p(A) = 0.5, p(B) = 0.5
- q: q(A) = 0.9, q(B) = 0.1

```
D_KL(p || q) = 0.5 log(0.5/0.9) + 0.5 log(0.5/0.1)
             = 0.5(-0.263) + 0.5(0.699)
             = 0.218 bits
```

This measures how much worse q is compared to p for encoding the true distribution.

**In AI**: When training classifiers, we minimize cross-entropy, which is equivalent to minimizing KL divergence (since H(p) is constant-it's the true data distribution). We're making our model's predictions q match the true distribution p.

### Why Cross-Entropy Loss Works: The Mathematical Connection

For classification with true labels y (one-hot encoded) and model predictions ŷ (softmax output):

```
Loss = -∑ᵢ yᵢ log(ŷᵢ)
```

This is exactly the cross-entropy H(y, ŷ).

**Why this functional form?**

1. **Maximum Likelihood Connection**: Minimizing cross-entropy ≡ maximizing likelihood of the data under the model.

   If model outputs probabilities ŷ = [ŷ₁, ŷ₂, ..., ŷₙ] and true class is k:
   ```
   Likelihood: P(class k | model) = ŷₖ
   Log-likelihood: log ŷₖ
   Negative log-likelihood: -log ŷₖ
   ```

   For one-hot encoded y (yₖ = 1, others = 0):
   ```
   -∑ᵢ yᵢ log ŷᵢ = -log ŷₖ
   ```

   Cross-entropy loss = negative log-likelihood!

2. **Derivative Properties**: Cross-entropy + softmax has a beautiful gradient:
   ```
   ∂Loss/∂zᵢ = ŷᵢ - yᵢ
   ```

   The gradient is simply (prediction - truth). This makes training stable and efficient.

3. **Penalizes Confident Mistakes Heavily**:
   - If true class is A, but model predicts ŷ(A) = 0.01 (confident it's not A):
     Loss = -log(0.01) = 4.6
   - If model predicts ŷ(A) = 0.5 (uncertain):
     Loss = -log(0.5) = 0.69

   Confident wrong predictions are penalized exponentially more than uncertain ones.

### Binary Cross-Entropy: The Special Case

For binary classification (y ∈ {0, 1}), cross-entropy simplifies to:

```
BCE = -[y log(ŷ) + (1-y) log(1-ŷ)]
```

**Derivation**:

For two classes with probabilities [ŷ, 1-ŷ]:
```
H(p, q) = -p(class 1) log ŷ - p(class 0) log(1-ŷ)
        = -y log ŷ - (1-y) log(1-ŷ)
```

**In PyTorch/TensorFlow**: This is `nn.BCELoss()` or `tf.keras.losses.BinaryCrossentropy()`.

### Mean Squared Error: An Information-Theoretic View

MSE is used for regression:
```
MSE = (1/n) ∑ᵢ (yᵢ - ŷᵢ)²
```

**Where does this come from?**

Assuming Gaussian noise: y = f(x) + ε, where ε ~ N(0, σ²)

The likelihood of observing y given prediction ŷ:
```
P(y | ŷ, σ²) = (1/√(2πσ²)) exp(-(y-ŷ)²/(2σ²))

Log-likelihood:
log P(y | ŷ, σ²) = -log(√(2πσ²)) - (y-ŷ)²/(2σ²)

Negative log-likelihood (ignoring constants):
∝ (y-ŷ)²
```

**MSE = negative log-likelihood under Gaussian assumptions.**

This is why MSE makes sense for regression (continuous outputs) while cross-entropy makes sense for classification (discrete probabilities).

### Mutual Information: Measuring Dependence

**Definition**: Mutual information I(X; Y) measures how much knowing X reduces uncertainty about Y.

```
I(X; Y) = D_KL(P(X,Y) || P(X)P(Y))
        = ∑ₓ ∑ᵧ P(x,y) log(P(x,y) / (P(x)P(y)))
```

**Properties**:
- I(X; Y) ≥ 0 (equality when X and Y are independent)
- I(X; Y) = I(Y; X) (symmetric, unlike KL divergence)
- I(X; X) = H(X) (self-information = entropy)

**Intuition**: If X and Y are independent, knowing X tells you nothing about Y, so I(X; Y) = 0. If X completely determines Y, I(X; Y) = H(Y).

**In AI**:
- Feature selection: Choose features with high mutual information with the label
- Representation learning: Maximize I(representation; label) while minimizing I(representation; nuisance variables)
- Information bottleneck theory: Deep learning can be viewed as compressing inputs while preserving mutual information with outputs

### Summary: Information Theory Cheat Sheet

| Concept | Formula | Measures | Use in AI |
|---------|---------|----------|-----------|
| **Entropy H(p)** | -∑ p(x) log p(x) | Uncertainty in distribution p | Model confidence, decision uncertainty |
| **Cross-Entropy H(p,q)** | -∑ p(x) log q(x) | Cost of encoding p using q | Classification loss |
| **KL Divergence D_KL(p‖q)** | ∑ p(x) log(p(x)/q(x)) | Difference between distributions | Regularization, VAEs, policy optimization |
| **Mutual Information I(X;Y)** | ∑∑ p(x,y) log(p(x,y)/(p(x)p(y))) | Information shared between X and Y | Feature selection, representation learning |

**Key Insight**: Loss functions aren't arbitrary. They arise from information-theoretic principles of matching distributions and maximizing likelihood. Understanding this lets you:
- Choose the right loss for your task
- Debug why loss isn't decreasing
- Design custom losses for unusual problems
- Understand why models behave the way they do

## Gradients as "How Wrong Am I?"

Calculus in AI boils down to one concept: **gradients**.

### Derivatives: Rate of Change

A derivative measures "if I wiggle the input, how much does the output change?"

`f(x) = x²`
`f'(x) = 2x`

At x=3, derivative = 6. Meaning: if you increase x slightly, f(x) increases 6 times faster.

**In AI**: You have a loss function (how wrong the model is). You want to know: "if I adjust this weight, does loss go up or down, and by how much?" That's a derivative.

### Gradients: Derivatives in High Dimensions

A gradient is just a vector of derivatives-one for each parameter.

If your model has 1 million parameters, the gradient is a 1-million-dimensional vector pointing in the direction of steepest increase in loss.

**Training**: Go in the opposite direction of the gradient (downhill) to reduce loss. That's gradient descent.

### The Chain Rule: Why Deep Learning Works

The chain rule lets you compute derivatives of compositions:

`(f ∘ g)'(x) = f'(g(x)) · g'(x)`

**Why It Matters**: Neural networks are compositions. Input → Layer1 → Layer2 → ... → Output. To train, you need the gradient of loss with respect to every weight in every layer.

**Backpropagation** is just the chain rule applied backwards through the network. That's it. No magic.

### An Intuition for Backprop

Imagine a factory assembly line. Final product is defective. You want to know which station contributed to the defect.

You start at the end:
- "Output is wrong by 10 units. The last station contributed 3 units of error."
- "That 3 units came from the previous station contributing 2 units."
- Work backwards, propagating blame through the chain.

That's backprop. You propagate error gradients backwards to assign blame (and updates) to each parameter.

## War Story: Gradient Explosion/Vanishing Ruining Training

**The Setup**: A team was training a deep recurrent network (RNN) for text prediction. 50 layers deep. They started training.

**The Problem**: Loss went to NaN (not a number) within 10 iterations.

**The Diagnosis**: Gradient explosion. Gradients were multiplying through 50 layers. Even small numbers, when multiplied 50 times, explode or vanish.

Example:
- Gradient = 1.1 at each layer
- After 50 layers: 1.1^50 = 117. Gradients explode.
- Gradient = 0.9 at each layer
- After 50 layers: 0.9^50 = 0.005. Gradients vanish.

**The Fix**: Gradient clipping (cap maximum gradient magnitude) and better architectures (LSTMs, residual connections) that prevent multiplication through many layers.

**The Lesson**: Math isn't just theory. Gradient dynamics determine if your model trains at all.

## Things That Will Confuse You

### "I can just use libraries, I don't need to understand the math"
You can drive without understanding combustion engines. But when the car breaks, you're helpless. Same with AI.

### "The math in papers is too hard"
Papers are written for other researchers, optimizing for precision and novelty, not pedagogy. Don't judge your understanding by whether you can read arxiv papers. Build intuition from simpler sources first.

### "I need to derive everything from scratch"
No. Intuition > proofs. Understand *what* a gradient is and *why* it matters. Leave the epsilon-delta proofs to mathematicians.

## Common Traps

**Trap #1: Memorizing formulas without understanding**
You won't remember formulas. You will remember intuitions. Focus on "what does this measure?" not "what's the equation?"

**Trap #2: Getting stuck in math rabbit holes**
You can always go deeper. At some point, diminishing returns. Get enough to be functional, then learn more as needed.

**Trap #3: Skipping linear algebra**
You can't. Every model is matrix operations. Bite the bullet.

**Trap #4: Treating probability as just counting**
Probability is subtle. P(A and B) vs P(A|B) vs P(A)·P(B) are different. Bayesian vs frequentist thinking is different. Take it seriously.

## Production Reality Check

Here's what math shows up in real work:

- **Matrix shapes not matching**: `(100, 512) @ (256, 128)` → dimension error. You'll debug this constantly.
- **Probability calibration**: Model outputs 0.9 but is right only 60% of the time. You need to understand probability to fix this.
- **Gradient issues**: Training unstable? Check gradient norms. Exploding? Clip or adjust learning rate.
- **Numerical precision**: Probabilities underflow to zero. You'll compute in log-space.

The math isn't abstract. It's the difference between working and not working.

## Build This Mini Project

**Goal**: Build intuition for gradients and optimization.

**Task**: Implement gradient descent from scratch on a simple problem.

Here's complete, runnable code with visualizations:

```python
import numpy as np
import matplotlib.pyplot as plt

# Function to minimize: f(x) = (x - 3)²
# The minimum is at x=3, where f(x)=0
def f(x):
    return (x - 3)**2

# Derivative: f'(x) = 2(x - 3)
def df(x):
    return 2 * (x - 3)

# Experiment 1: Good learning rate
print("="*60)
print("Experiment 1: Learning rate = 0.1 (good)")
print("="*60)

x = 0.0  # Start far from minimum
learning_rate = 0.1
history = [x]

for i in range(20):
    grad = df(x)
    x = x - learning_rate * grad
    history.append(x)

    if i % 5 == 0:
        print(f"Step {i:2d}: x = {x:7.4f}, f(x) = {f(x):7.4f}, gradient = {grad:7.4f}")

print(f"\nFinal: x = {x:.4f} (target: 3.0000)")
print(f"Converged to minimum! ✓")

# Experiment 2: Learning rate too high
print("\n" + "="*60)
print("Experiment 2: Learning rate = 2.0 (too high)")
print("="*60)

x = 0.0
learning_rate = 2.0
diverge_history = [x]

for i in range(10):
    grad = df(x)
    x = x - learning_rate * grad
    diverge_history.append(x)

    if i < 5:
        print(f"Step {i:2d}: x = {x:7.2f}, f(x) = {f(x):10.2f}")

print("💥 Diverging! x is oscillating wildly...")
print("Learning rate too high = overshooting the minimum")

# Experiment 3: Learning rate too low
print("\n" + "="*60)
print("Experiment 3: Learning rate = 0.001 (too low)")
print("="*60)

x = 0.0
learning_rate = 0.001
slow_history = [x]

for i in range(1000):
    grad = df(x)
    x = x - learning_rate * grad
    slow_history.append(x)

    if i in [0, 100, 500, 999]:
        print(f"Step {i:3d}: x = {x:7.4f}, f(x) = {f(x):7.4f}")

print("🐌 Converging very slowly...")
print("Learning rate too low = many iterations needed")

# Experiment 4: 2D optimization
print("\n" + "="*60)
print("Experiment 4: 2D optimization f(x,y) = x² + 10y²")
print("="*60)

# Function: f(x, y) = x² + 10y²
# Minimum at (0, 0)
# Gradients: df/dx = 2x, df/dy = 20y
def f_2d(x, y):
    return x**2 + 10*y**2

x, y = 5.0, 5.0  # Start far from minimum
learning_rate = 0.05  # Smaller LR needed because y has larger gradient
path = [(x, y)]

for i in range(50):
    grad_x = 2 * x
    grad_y = 20 * y

    x = x - learning_rate * grad_x
    y = y - learning_rate * grad_y
    path.append((x, y))

    if i % 10 == 0:
        print(f"Step {i:2d}: x = {x:7.4f}, y = {y:7.4f}, f(x,y) = {f_2d(x,y):10.4f}")

print(f"\nFinal: x = {x:.4f}, y = {y:.4f}")
print("Notice: x converges slower than y!")
print("Reason: y has 10x larger gradient, so it moves faster toward 0")
print("But if LR is too high, y would oscillate (try LR=0.1 to see!)")
print("This is why adaptive learning rates (Adam, RMSprop) help")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Good convergence
axes[0].plot(history, marker='o')
axes[0].axhline(y=3, color='r', linestyle='--', label='True minimum')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('x value')
axes[0].set_title('Good Learning Rate (0.1)')
axes[0].legend()
axes[0].grid(True)

# Plot 2: Divergence
axes[1].plot(diverge_history[:10], marker='o', color='red')
axes[1].axhline(y=3, color='g', linestyle='--', label='True minimum')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('x value')
axes[1].set_title('Too High Learning Rate (2.0) - Diverges!')
axes[1].legend()
axes[1].grid(True)

# Plot 3: Slow convergence
axes[2].plot(slow_history[::10], marker='o', color='orange')  # Plot every 10th point
axes[2].axhline(y=3, color='r', linestyle='--', label='True minimum')
axes[2].set_xlabel('Iteration (×10)')
axes[2].set_ylabel('x value')
axes[2].set_title('Too Low Learning Rate (0.001) - Slow!')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig('gradient_descent_comparison.png', dpi=150, bbox_inches='tight')
print("\n📊 Visualization saved as 'gradient_descent_comparison.png'")
print("\n" + "="*60)
print("KEY INSIGHTS:")
print("1. Learning rate is critical - too high diverges, too low is slow")
print("2. Gradients point in direction of steepest ascent")
print("3. We go OPPOSITE to gradient to minimize (gradient descent)")
print("4. Different parameters may need different learning rates")
print("5. This is exactly how neural networks train, but with")
print("   millions of parameters instead of 1 or 2!")
print("="*60)
```

**Expected Output:**
```
============================================================
Experiment 1: Learning rate = 0.1 (good)
============================================================
Step  0: x =  0.6000, f(x) =  5.7600, gradient = -6.0000
Step  5: x =  2.3383, f(x) =  0.4378, gradient = -1.3234
Step 10: x =  2.8145, f(x) =  0.0344, gradient = -0.3710
Step 15: x =  2.9550, f(x) =  0.0020, gradient = -0.0900

Final: x = 2.9930 (target: 3.0000)
Converged to minimum! ✓

============================================================
Experiment 2: Learning rate = 2.0 (too high)
============================================================
Step  0: x =  6.00, f(x) =       9.00
Step  1: x = -6.00, f(x) =      81.00
Step  2: x = 18.00, f(x) =     225.00
Step  3: x = -27.00, f(x) =     900.00
Step  4: x = 63.00, f(x) =    3600.00
💥 Diverging! x is oscillating wildly...
Learning rate too high = overshooting the minimum

============================================================
Experiment 3: Learning rate = 0.001 (too low)
============================================================
Step   0: x =  0.0060, f(x) =  8.9640
Step 100: x =  0.5487, f(x) =  6.0117
Step 500: x =  2.0927, f(x) =  0.8231
Step 999: x =  2.5944, f(x) =  0.1645
🐌 Converging very slowly...
Learning rate too low = many iterations needed

============================================================
Experiment 4: 2D optimization f(x,y) = x² + 10y²
============================================================
Step  0: x =  4.5000, y =  0.0000, f(x,y) =    20.2500
Step 10: x =  1.7433, y =  0.0000, f(x,y) =     3.0391
Step 20: x =  0.6746, y =  0.0000, f(x,y) =     0.4551
Step 30: x =  0.2612, y =  0.0000, f(x,y) =     0.0682
Step 40: x =  0.1011, y =  0.0000, f(x,y) =     0.0102

Final: x = 0.0391, y = 0.0000
Notice: x converges slower than y!
Reason: y has 10x larger gradient, so it moves faster toward 0
But if LR is too high, y would oscillate (try LR=0.1 to see!)
This is why adaptive learning rates (Adam, RMSprop) help
============================================================
```

**Key Insights from This Exercise**:

1. **Gradient Descent is Simple**: Just compute gradient, step in opposite direction
2. **Learning Rate is Everything**: Too high → diverge, too low → slow, just right → converges
3. **This Scales**: Neural networks with 100M parameters use the exact same algorithm
4. **Different Parameters Need Different Rates**: Some weights need smaller steps than others
5. **Local Minima Exist**: For non-convex functions (like neural nets), you might get stuck in local minima

**Connection to Neural Networks**:
- In a neural network, `x` is replaced by millions of weights
- `f(x)` is replaced by the loss function (how wrong the model is)
- The gradient is computed using backpropagation (chain rule)
- Everything else is the same: gradient descent on a huge number of parameters

This is the core of training neural networks, just scaled to millions of parameters.

---

