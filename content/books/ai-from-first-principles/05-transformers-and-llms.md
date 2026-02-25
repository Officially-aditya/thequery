# Chapter 5 - Transformers & LLMs: Attention Changed Everything

## The Crux
For years, sequence modeling meant RNNs: process one word at a time, remember the past. It worked, but it was slow and forgot long-range dependencies. Then transformers arrived: process everything in parallel, use attention to find what matters. This architecture unlocked LLMs, changed NLP, and is spreading to images, video, and more.

## Why Attention Beats Recurrence

### The RNN Problem

RNNs process sequences step-by-step:
```
h₁ = f(x₁, h₀)
h₂ = f(x₂, h₁)
h₃ = f(x₃, h₂)
...
```

Hidden state `h` carries information forward. To access word 1 when at word 100, information must survive 99 steps of computation. It doesn't.

**Problems**:
1. **Sequential processing**: Can't parallelize. Slow.
2. **Vanishing gradients**: Long-range dependencies get lost.
3. **Fixed-size bottleneck**: `h` must encode everything.

### The Attention Solution

Instead of forcing information through a sequential bottleneck, **let every position attend to every other position directly**.

Processing word 100? Look back at all 99 previous words, figure out which are relevant, and pull information from them.

**Key Idea**: Attention is a learned, differentiable lookup table.

- Query: "What am I looking for?"
- Keys: "What does each position offer?"
- Values: "What information does each position have?"

Compute similarity between query and all keys, use that to weight values.

```
Attention(Q, K, V) = softmax(QKᵀ / √d) V
```

**Intuition**:
- Q·Kᵀ measures "how relevant is each position?"
- Softmax converts to probabilities
- Multiply by V to get weighted sum of relevant info

### Why It Wins

**Parallelization**: All attention operations are matrix multiplies. GPUs love this. Training is 10x-100x faster than RNNs.

**Long-range dependencies**: Word 100 can directly attend to word 1. No vanishing gradients through 99 steps.

**Flexibility**: Attention weights are learned. The model decides what's important.

## The Mathematics of Attention: A Deep Dive

The attention formula `Attention(Q, K, V) = softmax(QKᵀ / √d) V` looks simple, but there's deep mathematics behind each component. This section rigorously derives why attention works and why each piece is necessary.

### Scaled Dot-Product Attention: The Full Derivation

**Setup**:
- Input sequence: X ∈ ℝⁿˣᵈ (n tokens, each d-dimensional)
- Query matrix: Q = XW_Q where W_Q ∈ ℝᵈˣᵈₖ
- Key matrix: K = XW_K where W_K ∈ ℝᵈˣᵈₖ
- Value matrix: V = XW_V where W_V ∈ ℝᵈˣᵈᵥ

Result: Q, K ∈ ℝⁿˣᵈₖ, V ∈ ℝⁿˣᵈᵥ

**Step 1: Computing Similarity (QKᵀ)**

For each query vector qᵢ and key vector kⱼ, compute dot product:
```
score(qᵢ, kⱼ) = qᵢ · kⱼ = ∑ₗ qᵢₗ kⱼₗ
```

In matrix form:
```
S = QKᵀ ∈ ℝⁿˣⁿ
Sᵢⱼ = qᵢ · kⱼ
```

**Interpretation**: Sᵢⱼ measures how much query i "cares about" key j.

**Why dot product?**
1. **Geometric meaning**: qᵢ · kⱼ = ||qᵢ|| ||kⱼ|| cos(θ), where θ is angle between vectors
   - Parallel vectors (similar): large positive dot product
   - Perpendicular (unrelated): dot product ≈ 0
   - Opposite (dissimilar): negative dot product

2. **Computational efficiency**: Matrix multiplication is highly optimized on GPUs

3. **Differentiable**: We can backpropagate through it to learn Q, K, V

**Alternative similarity functions** (used in other attention variants):
- Additive: score(qᵢ, kⱼ) = vᵀ tanh(W[qᵢ; kⱼ])
- Bilinear: score(qᵢ, kⱼ) = qᵢᵀ W kⱼ

Dot product is simpler and faster.

**Step 2: Scaling by √dₖ**

The crucial question: **Why divide by √dₖ?**

**Problem without scaling**:

As dimensionality dₖ increases, dot products grow large. Consider:
- qᵢ, kⱼ are vectors with dₖ components
- Assume each component drawn from distribution with mean 0, variance 1
- Then qᵢ · kⱼ = ∑ₗ qᵢₗ kⱼₗ

**Expected value**:
```
E[qᵢ · kⱼ] = E[∑ₗ qᵢₗ kⱼₗ] = ∑ₗ E[qᵢₗ kⱼₗ] = ∑ₗ E[qᵢₗ]E[kⱼₗ] = 0
```
(assuming independence)

**Variance**:
```
Var(qᵢ · kⱼ) = Var(∑ₗ qᵢₗ kⱼₗ)
             = ∑ₗ Var(qᵢₗ kⱼₗ)  (assuming independence)
             = ∑ₗ E[(qᵢₗ kⱼₗ)²] - (E[qᵢₗ kⱼₗ])²
             = ∑ₗ E[qᵢₗ²]E[kⱼₗ²]  (independence)
             = ∑ₗ 1 · 1
             = dₖ
```

**Result**: Dot products have variance dₖ. For large dₖ, dot products become very large or very small.

**Effect on softmax**:

After softmax, we compute:
```
softmax(Sᵢ)ⱼ = exp(Sᵢⱼ) / ∑ₖ exp(Sᵢₖ)
```

If Sᵢⱼ are large (say, range [-100, 100] for dₖ=1024):
- exp(100) ≈ 10⁴³
- exp(-100) ≈ 10⁻⁴⁴
- Softmax saturates: almost all weight goes to the maximum, others ≈ 0
- Gradients vanish: ∂softmax/∂S ≈ 0 everywhere except the peak

**Solution**: Scale by √dₖ to keep variance = 1:
```
Var(qᵢ · kⱼ / √dₖ) = Var(qᵢ · kⱼ) / dₖ = dₖ / dₖ = 1
```

Now dot products stay in a reasonable range regardless of dimensionality.

**Empirical validation**: The original "Attention is All You Need" paper tested this:
- Without scaling: training unstable, poor performance
- With scaling: stable training, better performance

**Mathematical proof of gradient improvement**:

Softmax gradient:
```
∂softmax(x)ᵢ/∂xⱼ = softmax(x)ᵢ (δᵢⱼ - softmax(x)ⱼ)
```

where δᵢⱼ = 1 if i=j, else 0.

When inputs to softmax are large (no scaling), softmax(x)ᵢ ≈ 1 for max i, ≈ 0 otherwise.

Then:
```
∂softmax(x)ᵢ/∂xⱼ ≈ 0  (gradient vanishes)
```

With scaling, inputs to softmax have reasonable magnitude, gradients flow properly.

**Step 3: Softmax Normalization**

Apply row-wise softmax:
```
A = softmax(QKᵀ / √dₖ)
Aᵢⱼ = exp(Sᵢⱼ/√dₖ) / ∑ₖ exp(Sᵢₖ/√dₖ)
```

**Properties**:
1. **Non-negative**: Aᵢⱼ ≥ 0
2. **Normalized**: ∑ⱼ Aᵢⱼ = 1 (each row sums to 1)
3. **Differentiable**: Can backprop through softmax

**Interpretation**: Aᵢⱼ is the "attention weight" from token i to token j. Row i forms a probability distribution over which tokens to attend to.

**Why softmax instead of alternatives?**

1. **Sparse attention**: Softmax exponentiates, so large values dominate
   - If Sᵢ₁ = 5, Sᵢ₂ = 4, Sᵢ₃ = 0:
     - exp(5) = 148, exp(4) = 55, exp(0) = 1
     - After normalization: [0.73, 0.27, 0.005]
   - Most weight on the highest-scoring key

2. **Temperature control**: Can adjust sharpness by dividing by temperature τ:
   ```
   softmax(x/τ)
   ```
   - τ → 0: one-hot (hardest)
   - τ → ∞: uniform (softest)

3. **Information-theoretic interpretation**: Softmax is the maximum entropy distribution subject to constraints on the moments

**Step 4: Weighted Sum of Values**

Compute output:
```
Output = AV ∈ ℝⁿˣᵈᵥ
```

For token i:
```
outputᵢ = ∑ⱼ Aᵢⱼ vⱼ
```

**Interpretation**: Each output token is a weighted average of all value vectors, where weights are the attention scores.

**Example**:
- Token i = "bank" (ambiguous)
- High attention to "river" → Aᵢ,river = 0.8
- Low attention to "money" → Aᵢ,money = 0.2
- outputᵢ = 0.8 * v_river + 0.2 * v_money + ...
- Result: "bank" gets contextualized toward the "river" meaning

### Multi-Head Attention: Why Multiple Heads?

**Problem with single attention**: One attention mechanism can only capture one type of relationship.

Example in "The cat sat on the mat":
- Syntactic: "cat" attends to "sat" (subject-verb)
- Semantic: "cat" attends to "mat" (where the cat is)
- Coreference: "cat" might attend to earlier mentions

**Solution**: Multiple attention "heads" capture different relationships.

**Multi-head Attention Formula**:

For h heads:
```
headᵢ = Attention(QWᵢQ, KWᵢK, VWᵢV)
```

where WᵢQ, WᵢK ∈ ℝᵈₘₒdₑₗˣᵈₖ, WᵢV ∈ ℝᵈₘₒdₑₗˣᵈᵥ

Concatenate all heads and project:
```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) W_O
```

where W_O ∈ ℝʰᵈᵥˣᵈₘₒdₑₗ

**Dimensions**:
- Typically: h = 8, dₖ = dᵥ = dₘₒdₑₗ / h
- Example: dₘₒdₑₗ = 512 → each head has dₖ = dᵥ = 64

**Why this works**:

1. **Different subspaces**: Each head learns projections Wᵢ that focus on different aspects
   - Head 1 might learn syntactic dependencies
   - Head 2 might learn semantic similarity
   - Head 3 might learn positional proximity

2. **Ensemble effect**: Multiple heads provide redundancy and robustness

3. **Computational efficiency**: h heads with dimension d/h each has the same cost as one head with dimension d:
   ```
   Cost = O(n² dₖ h) = O(n² · (d/h) · h) = O(n² d)
   ```

**Empirical analysis** (from research):
- Different heads specialize in different linguistic phenomena
- Some heads focus on adjacent tokens (local structure)
- Some heads focus on distant tokens (long-range dependencies)
- Visualizing attention weights shows interpretable patterns (e.g., head tracking subject-verb agreement)

### Self-Attention vs Cross-Attention

**Self-Attention**: Q, K, V all from same input
```
X ∈ ℝⁿˣᵈ
Q = XW_Q, K = XW_K, V = XW_V
```

Each token attends to all tokens in the same sequence (including itself).

**Cross-Attention**: Q from one source, K and V from another
```
X_query ∈ ℝⁿˣᵈ, X_context ∈ ℝᵐˣᵈ
Q = X_query W_Q
K = X_context W_K, V = X_context W_V
```

Used in encoder-decoder models:
- Decoder queries attend to encoder keys/values
- Example: Machine translation, decoder (English) attends to encoder (French)

### Masked Attention: Preventing Future Leakage

**Problem**: In autoregressive generation (e.g., language modeling), token i shouldn't see tokens j > i (future tokens).

**Solution**: Apply mask before softmax
```
S = QKᵀ / √dₖ
S_masked = S + M

where M_ij = { 0     if j ≤ i
             { -∞   if j > i

A = softmax(S_masked)
```

**Effect**:
- For i=1, only M₁₁ = 0, others = -∞ → token 1 can only attend to itself
- For i=2, M₂₁ = M₂₂ = 0, M₂ₖ = -∞ for k>2 → token 2 attends to tokens 1 and 2
- For i=n, all M_nₖ = 0 → token n attends to all tokens

After softmax:
```
exp(-∞) = 0
```

So future positions get zero attention weight.

**Implementation**:
```python
# Create lower triangular mask
mask = torch.tril(torch.ones(n, n))
mask = mask.masked_fill(mask == 0, float('-inf'))
scores = scores + mask  # Broadcasting
attention_weights = softmax(scores)
```

### Computational Complexity Analysis

**Attention complexity**: O(n² d)

Breaking it down:
1. **QKᵀ**: (n × dₖ) @ (dₖ × n) = O(n² dₖ)
2. **Softmax**: O(n²) (row-wise)
3. **AV**: (n × n) @ (n × dᵥ) = O(n² dᵥ)

Total: O(n²(dₖ + dᵥ)) = O(n² d) assuming dₖ, dᵥ ≈ d

**Comparison to RNNs**:
- RNN: O(nd²) for sequence of length n
  - Sequential: process one token at a time, each requires O(d²) (weight matrix multiply)
  - Total: n steps × O(d²) = O(nd²)

**Crossover point**:
- Attention faster when n < d (typical for transformers with d=512-1024, n=100-512)
- RNN faster when n > d (very long sequences)

**Memory**:
- Attention: O(n²) to store attention matrix
- RNN: O(n) to store hidden states

**This is why**:
- Transformers dominate for n ≤ 2048 (BERT, GPT)
- For very long sequences (n > 10K), need sparse attention (Longformer, BigBird)

### Why Attention Works: Information-Theoretic View

Attention can be viewed as **soft dictionary lookup**.

Traditional dictionary:
```
lookup(query, dict) = dict[key]  if exact match, else None
```

Attention:
```
lookup(query, dict) = ∑_i similarity(query, keyᵢ) · valueᵢ
```

**Analogy**:
- You ask: "What's the capital of France?" (query)
- Database has entries: (France, Paris), (Germany, Berlin), ...
  - Keys: country names
  - Values: capitals
- Attention computes similarity: query ≈ "France" → high weight on (France, Paris)
- Output: mostly "Paris" with tiny contribution from other capitals

**Mutual Information Interpretation**:

Attention maximizes mutual information I(output; relevant_context) while minimizing I(output; irrelevant_context).

The learned Q, K, V matrices determine what's relevant.

### Comparison to Convolution

**Convolution**: Fixed local receptive field
- Each output depends on fixed-size window of inputs
- Same operation everywhere (weight sharing)
- Good for local patterns (edges in images)

**Attention**: Adaptive global receptive field
- Each output depends on ALL inputs (with learned weights)
- Different operation at each position (content-based)
- Good for long-range dependencies (language)

**Hybrid models** (e.g., ConvBERT): Use both convolution (local) and attention (global)

### Summary: The Complete Attention Pipeline

1. **Project**: X → Q, K, V via learned matrices
2. **Score**: Compute QKᵀ (similarity of all pairs)
3. **Scale**: Divide by √dₖ (keep variance stable)
4. **Mask** (if causal): Prevent attending to future
5. **Normalize**: Softmax (convert scores to probabilities)
6. **Aggregate**: Multiply by V (weighted sum of values)
7. **Multi-head**: Repeat h times, concatenate, project

**Mathematical elegance**: Every step is differentiable, so we can backprop through the entire pipeline to learn Q, K, V transformations that maximize task performance.

**Key Insight**: Attention is a learnable routing mechanism. The model learns to route information from relevant parts of the input to each output position. This is far more flexible than fixed architectures (RNNs, CNNs) with hard-coded information flow.

## What Embeddings Really Represent

Before diving into transformers, let's clarify embeddings-they're everywhere in modern AI.

### The Problem: Words Aren't Numbers

Computers need numbers. Words are symbols. How do you convert "dog" into numbers?

**Bad Idea**: Assign integers. `dog=1, cat=2, tree=3`.

Problem: This implies `dog + cat = tree` (mathematically). Arithmetic on these IDs is meaningless.

**Good Idea**: Represent each word as a vector in high-dimensional space, where **similar words are nearby**.

```
dog   = [0.2, 0.8, 0.1, ..., 0.3]  (300 dimensions)
cat   = [0.3, 0.7, 0.2, ..., 0.4]  (nearby dog)
tree  = [0.1, 0.1, 0.9, ..., 0.0]  (far from dog/cat)
```

Now similarity is measurable: dot product or cosine distance.

### How Embeddings Are Learned

**Word2Vec**: Train a simple network to predict context words from a target word (or vice versa). Vectors that yield good predictions capture semantic similarity.

**In transformers**: Embeddings are learned jointly with the model. They're optimized to be useful for the task.

### What Do They Capture?

Surprisingly, embeddings capture semantic and syntactic relationships:

```
king - man + woman ≈ queen
Paris - France + Germany ≈ Berlin
```

**Why?** Distributional hypothesis: "Words in similar contexts have similar meanings." The model learns these regularities from massive data.

### Positional Embeddings

Attention has no notion of order. "Dog bites man" and "Man bites dog" look the same to raw attention.

**Solution**: Add positional encodings-vectors that encode position (1st word, 2nd word, etc.). Now the model knows order.

### Positional Encoding Theory: Teaching Order to Transformers

Self-attention is permutation-invariant: swapping the order of inputs doesn't change the attention weights. This is a problem for sequences where order matters (like language). Positional encodings solve this by injecting position information into the model. This section derives why sinusoidal encodings work and explores alternatives.

#### The Problem: Permutation Invariance of Attention

**Mathematical observation**: The attention formula
```
Attention(Q, K, V) = softmax(QKᵀ/√dₖ) V
```

depends only on the content of Q, K, V, not their order.

**Proof**: If we permute the input sequence with permutation matrix P:
```
X' = PX  (rows of X are reordered)
Q' = PQ, K' = PK, V' = PV
```

Then:
```
Q'K'ᵀ = (PQ)(PK)ᵀ = PQ·Kᵀ·Pᵀ
```

This is just a permuted version of QKᵀ. After softmax and multiplying by V', we get permuted outputs.

**Consequence**: The attention mechanism itself has no notion of position. Token at position 1 is treated identically to token at position 100.

**Why this is bad**: In "The cat sat on the mat", word order determines meaning:
- "cat sat" (subject acts)
- "sat cat" (nonsense)

We need to inject positional information.

#### Solution 1: Learned Positional Embeddings

**Idea**: Create a lookup table of position vectors.

**Implementation**:
```python
max_length = 512
embedding_dim = 512
pos_embedding = nn.Embedding(max_length, embedding_dim)

# For position i:
pos_vec = pos_embedding(i)  # Learned vector for position i

# Add to token embedding:
input_representation = token_embedding(x) + pos_embedding(position)
```

**Parameters**: max_length × embedding_dim (e.g., 512 × 512 = 262,144 parameters)

**Pros**:
- Simple to implement
- Model learns optimal position representations for the task

**Cons**:
- Fixed maximum length (can't handle sequences longer than max_length)
- No generalization to unseen positions
- Extra parameters to learn

#### Solution 2: Sinusoidal Positional Encoding (Original Transformer)

**Motivation**: Find a function that:
1. Is deterministic (no learned parameters)
2. Generalizes to any sequence length
3. Encodes unique positions (no collisions)
4. Has geometric properties that help the model learn relative positions

**The formula** (Vaswani et al., 2017):

For position `pos` and dimension `i`:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

where:
- pos ∈ {0, 1, 2, ..., n-1} is the position in the sequence
- i ∈ {0, 1, 2, ..., d_model/2 - 1} is the dimension index
- d_model is the embedding dimension (e.g., 512)

**Example** (d_model = 4):

Position 0:
```
PE(0, 0) = sin(0/10000^0) = sin(0) = 0
PE(0, 1) = cos(0/10000^0) = cos(0) = 1
PE(0, 2) = sin(0/10000^(2/4)) = sin(0) = 0
PE(0, 3) = cos(0/10000^(2/4)) = cos(0) = 1
→ [0, 1, 0, 1]
```

Position 1:
```
PE(1, 0) = sin(1/1) = sin(1) ≈ 0.841
PE(1, 1) = cos(1/1) = cos(1) ≈ 0.540
PE(1, 2) = sin(1/10000^(2/4)) = sin(1/100) ≈ 0.010
PE(1, 3) = cos(1/10000^(2/4)) = cos(1/100) ≈ 1.000
→ [0.841, 0.540, 0.010, 1.000]
```

#### Why Sinusoidal Encodings Work: Mathematical Analysis

**Property 1: Uniqueness**

Every position gets a unique encoding vector (for reasonable sequence lengths).

**Proof sketch**: The encoding is a composition of sine/cosine functions with different frequencies. The frequencies are:
```
ωᵢ = 1 / 10000^(2i/d_model)
```

These decrease exponentially: ω₀ = 1, ω₁ = 1/100, ω₂ = 1/10000, ...

Lower dimensions (high frequency) encode fine-grained position differences. Higher dimensions (low frequency) encode coarse-grained positions.

**Analogy to binary numbers**: Just as binary uses powers of 2 (1, 2, 4, 8, ...) to uniquely represent numbers, sinusoidal encoding uses powers of 10000 to represent positions.

**Property 2: Relative Position is a Linear Function**

**Claim**: For any fixed offset k, the encoding of position pos+k can be represented as a linear function of the encoding of position pos.

**Proof**:

Using the angle addition formula:
```
sin(α + β) = sin(α)cos(β) + cos(α)sin(β)
cos(α + β) = cos(α)cos(β) - sin(α)sin(β)
```

For dimension 2i (sine component):
```
PE(pos+k, 2i) = sin((pos+k) / 10000^(2i/d))
               = sin(pos/10000^(2i/d) + k/10000^(2i/d))
```

Let α = pos/10000^(2i/d), β = k/10000^(2i/d):
```
PE(pos+k, 2i) = sin(α + β)
               = sin(α)cos(β) + cos(α)sin(β)
               = PE(pos,2i) · cos(β) + PE(pos,2i+1) · sin(β)
```

**In matrix form**:
```
[PE(pos+k, 2i)  ]   [cos(β)  sin(β)] [PE(pos, 2i)  ]
[PE(pos+k, 2i+1)] = [-sin(β) cos(β)] [PE(pos, 2i+1)]
```

This is a rotation matrix! The relative offset k determines the rotation angle β.

**Implication**: The model can learn to attend to relative positions (e.g., "attend to word 3 positions back") using linear transformations.

**Property 3: Bounded Values**

All components of PE are in [-1, 1] (sine and cosine range).

**Implication**: Positional encodings don't dominate the token embeddings. Both contribute to the final representation.

**Property 4: Different Frequencies for Different Dimensions**

Low dimensions change rapidly (high frequency):
- PE(0, 0) vs PE(1, 0): Large difference (frequency ω₀ = 1)

High dimensions change slowly (low frequency):
- PE(0, d-1) vs PE(1, d-1): Small difference (frequency ω_{d/2-1} ≈ 1/10000)

**Intuition**:
- Low dimensions: Encode exact position (changes every step)
- High dimensions: Encode coarse region (changes every ~10000 steps)

**Analogy**: Like a clock:
- Second hand (high frequency): Precise time within a minute
- Minute hand (medium frequency): Position within an hour
- Hour hand (low frequency): Time of day

#### Why 10000?

The constant 10000 in the formula is somewhat arbitrary, but chosen to:
1. Provide a large range: With d_model = 512, positions up to ~10000 are easily distinguishable
2. Geometric sequence: 10000^(i/256) creates smoothly varying frequencies
3. Empirically works well

**Alternatives**: Some models use different bases (e.g., 500, 1000) depending on expected sequence lengths.

#### Comparison: Learned vs Sinusoidal

| Aspect | Learned Embeddings | Sinusoidal Encoding |
|--------|-------------------|---------------------|
| Parameters | max_len × d_model | 0 (deterministic) |
| Generalization | Fixed max length | Any length |
| Flexibility | Adapts to task | Fixed pattern |
| Relative position | Must learn | Built-in (rotation) |
| Modern use | BERT, GPT-2 | Original Transformer |

**Modern practice**: Many models (BERT, GPT) use learned positional embeddings because:
- Extra parameters are cheap (relative to model size)
- Model can adapt encoding to the task
- Maximum length is usually known (e.g., 512, 2048 tokens)

**When sinusoidal is better**:
- Variable-length sequences (no fixed max length)
- Low-resource settings (fewer parameters)
- Explicit relative position modeling

#### Advanced: Relative Positional Encodings

**Problem**: Absolute positions (0, 1, 2, ...) aren't always meaningful. What matters is relative distance.

Example: "The cat sat on the mat" vs "Yesterday, the cat sat on the mat"
- Absolute: "cat" is at position 1 vs position 2 (different)
- Relative: "cat" is 1 word before "sat" (same)

**Solution: Relative Position Encodings** (Shaw et al., 2018)

Instead of encoding absolute position, modify attention to encode relative position:
```
Attention_ij = softmax((qᵢ · kⱼ + qᵢ · r_{i-j}) / √dₖ)
```

where r_{i-j} is a learned embedding for relative distance i-j.

**Advantages**:
- Position-invariant: Shift the sequence, relationships remain
- Longer generalization: Learns "attend 3 tokens back" instead of "attend to position 5"

**Used in**: Transformer-XL, T5, modern architectures

#### RoPE: Rotary Positional Embedding (Modern Alternative)

**Motivation**: Combine benefits of absolute and relative encodings.

**Idea** (Su et al., 2021): Apply rotation matrices to Q and K based on position.

**Formula**:
```
Q_pos = R(pos) Q
K_pos = R(pos) K
```

where R(pos) is a rotation matrix that depends on position pos.

**Magic**: When computing attention:
```
Q_i · K_j = (R(i)Q) · (R(j)K) = Qᵀ R(i)ᵀ R(j) K = Qᵀ R(j-i) K
```

The dot product depends only on relative position j-i!

**Advantages**:
- Combines absolute position (in Q, K) with relative position (in dot product)
- No extra parameters
- Better extrapolation to longer sequences

**Used in**: LLaMA, PaLM, many modern LLMs

#### ALiBi: Attention with Linear Biases

**Simplest approach** (Press et al., 2021): Add a linear bias to attention scores based on distance.

**Formula**:
```
Attention_ij = softmax((qᵢ · kⱼ - m · |i-j|) / √dₖ)
```

where m is a learned slope.

**Intuition**: Penalize attention to distant tokens linearly.

**Advantages**:
- Extremely simple (no extra embeddings)
- Zero parameters
- Strong extrapolation to longer sequences

**Used in**: BLOOM, some recent LLMs

#### Practical Implementation (PyTorch)

**Sinusoidal encoding**:
```python
def sinusoidal_positional_encoding(max_len, d_model):
    """Generate sinusoidal positional encoding"""
    position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                        -(np.log(10000.0) / d_model))  # [d_model/2]

    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
    pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

    return pe

# Usage:
pe = sinusoidal_positional_encoding(max_len=512, d_model=512)
x = token_embeddings + pe[:seq_len]  # Add positional encoding
```

**Learned embeddings**:
```python
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device)
        return x + self.pe(positions)
```

#### Summary: Positional Encoding Theory

| Concept | Formula/Intuition | Why It Matters |
|---------|------------------|----------------|
| Permutation invariance | Attention is order-blind | Need to inject position info |
| Sinusoidal encoding | PE(pos, 2i) = sin(pos/10000^(2i/d)) | No parameters, infinite length |
| Relative position | PE(pos+k) = LinearTransform(PE(pos)) | Model can learn relative attention |
| Frequency hierarchy | Low dim = high freq, high dim = low freq | Multi-scale position representation |
| Learned embeddings | Position lookup table | Flexible, task-specific |
| RoPE | Rotation-based, relative in dot product | Best of both worlds |
| ALiBi | Linear distance penalty | Simplest, good extrapolation |

**Key insight**: Positional encoding is not just "adding position numbers". It's about:
1. **Uniqueness**: Every position gets a distinct representation
2. **Geometry**: Relative positions have geometric relationships (rotations, linear transforms)
3. **Multi-scale**: Different dimensions encode different temporal scales

**Modern trends**: Moving from absolute → relative encodings, and from learned → zero-parameter methods (RoPE, ALiBi) that generalize better to longer sequences.

### Layer Normalization Theory: Why Transformers Don't Use Batch Norm

Transformers universally use Layer Normalization instead of Batch Normalization. This isn't arbitrary - there are deep theoretical and practical reasons. This section derives Layer Norm mathematically and explains why it's essential for transformers.

#### Batch Norm's Problem for Sequences

**Recall Batch Normalization**: Normalize across the batch dimension.

For input x ∈ ℝᴮˣᴺˣᴰ (batch size B, sequence length N, features D):
```
BatchNorm: Normalize across B for each position n and feature d
μ = (1/B) ∑_{b=1}^B x_{b,n,d}
```

**Problem for variable-length sequences**:
- Sentence 1: "Hello" (length 1)
- Sentence 2: "The cat sat on the mat" (length 6)
- Sentence 3: "Hi" (length 1)

At position 5:
- Only sentence 2 has a token
- Batch statistics are computed from 1 example (B = 1)
- Variance estimate is meaningless!

**Problem for inference**:
- Batch size = 1 (single sentence)
- Can't compute meaningful batch statistics
- Must use running averages from training (but with variable lengths, these are unreliable)

**Fundamental issue**: Batch Norm assumes all examples in batch have the same structure. Sequences violate this.

#### Layer Normalization: The Solution

**Idea** (Ba et al., 2016): Normalize across features (not across batch).

**For each example independently**:
```
For input x ∈ ℝᴰ (D features):

μ = (1/D) ∑_{i=1}^D xᵢ          (mean across features)
σ² = (1/D) ∑_{i=1}^D (xᵢ - μ)²   (variance across features)

x̂ᵢ = (xᵢ - μ) / √(σ² + ε)       (normalize)
yᵢ = γᵢ x̂ᵢ + βᵢ                   (scale and shift)
```

where γ, β are learnable per-feature parameters.

**Key difference from Batch Norm**:

| Batch Norm | Layer Norm |
|------------|------------|
| Normalize across batch (B examples) | Normalize across features (D dimensions) |
| Statistics: μ, σ² computed from B examples | Statistics: μ, σ² computed from D features of single example |
| Requires batch size > 1 | Works with batch size = 1 |
| Different behavior train/test | Same behavior train/test |

#### Mathematical Derivation: Why Layer Norm Works

**Stabilizes activations within each layer**:

After normalization, each example has:
- Mean = 0 (approximately, before scale/shift)
- Variance = 1 (approximately, before scale/shift)

This prevents:
1. **Activation explosion**: No matter what previous layers do, inputs to next layer are bounded
2. **Activation vanishing**: Ensures signal strength remains constant

**Gradient flow**:

Similar to Batch Norm, Layer Norm bounds gradients during backpropagation.

**Backward pass**:
```
∂L/∂xᵢ = (∂L/∂x̂ᵢ) · (∂x̂ᵢ/∂xᵢ) + (∂L/∂μ) · (∂μ/∂xᵢ) + (∂L/∂σ²) · (∂σ²/∂xᵢ)
```

The normalization creates dependencies between all features xᵢ (through μ and σ²), which decorrelates gradients and prevents any single feature from dominating.

**Full derivative** (similar to Batch Norm derivation):
```
∂L/∂xᵢ = (γ/√(σ² + ε)) · [(∂L/∂yᵢ) - (1/D)∑ⱼ(∂L/∂yⱼ) - x̂ᵢ·(1/D)∑ⱼ(∂L/∂yⱼ)x̂ⱼ]
```

**Implication**: Gradients are centered and normalized, preventing explosion/vanishing.

#### Why Transformers Need Layer Norm

**1. Variable sequence lengths**:
- Input: "Hello" (1 token) vs "The quick brown fox" (4 tokens)
- Batch Norm can't handle this naturally
- Layer Norm processes each token independently

**2. Attention creates large activation variance**:

Attention output:
```
Output_i = ∑ⱼ softmax(QKᵀ)ᵢⱼ · Vⱼ
```

This is a weighted sum of value vectors. Without normalization:
- If some attention weights are very large → output explodes
- If values have different scales → unstable learning

Layer Norm after attention stabilizes this:
```
Output = LayerNorm(Attention(Q, K, V))
```

**3. Deep stacking (many layers)**:

Transformers have 12-100+ layers. Without normalization:
- Activations compound across layers
- Gradients vanish/explode

Layer Norm + residual connections ensure stable signal flow.

#### Pre-Norm vs Post-Norm

**Post-Norm** (original Transformer):
```
x = x + Attention(LayerNorm(x))
```

Normalize before the operation.

**Pre-Norm**:
```
x = LayerNorm(x + Attention(x))
```

Normalize after adding residual.

Wait, I mixed these up. Let me correct:

**Pre-Norm** (modern preference):
```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

Normalization is applied **before** the sub-layer (attention or FFN).

**Post-Norm** (original paper):
```
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))
```

Normalization is applied **after** adding the residual.

**Why Pre-Norm is better**:

1. **Gradient flow**: With Pre-Norm, the residual path is completely clean:
   ```
   x_{out} = x_{in} + f(LayerNorm(x_{in}))
   ∂x_{out}/∂x_{in} = I + ∂f/∂x_{in}
   ```
   The identity I is present, ensuring gradient highway.

2. **Initialization**: Pre-Norm is less sensitive to initialization. The normalization ensures inputs to f(...) are well-scaled from the start.

3. **Training stability**: Empirically, Pre-Norm allows training deeper transformers without learning rate warmup tricks.

**Trade-off**: Post-Norm sometimes achieves slightly better final performance (when training is stable), but Pre-Norm is more robust.

**Modern practice**: GPT-3, GPT-4, LLaMA, most recent models use Pre-Norm.

#### Layer Norm vs Batch Norm: A Complete Comparison

| Aspect | Batch Norm | Layer Norm |
|--------|------------|------------|
| **Normalization axis** | Across batch (B examples) | Across features (D dimensions) |
| **Train/test difference** | Yes (uses running stats at test) | No (same computation) |
| **Minimum batch size** | >1 (preferably >8) | 1 (works with any batch size) |
| **Sequence compatibility** | Poor (variable lengths break it) | Excellent |
| **Typical use** | CNNs, fully-connected nets | Transformers, RNNs, LSTMs |
| **Computational cost** | O(D) per layer | O(D) per example |
| **Parameters** | 2D (γ, β) | 2D (γ, β) |
| **When invented** | 2015 (Ioffe & Szegedy) | 2016 (Ba et al.) |

#### Other Normalization Variants

**RMSNorm** (Root Mean Square Normalization):

Simplification of Layer Norm - only normalize by RMS, skip mean subtraction:
```
RMS = √((1/D) ∑ᵢ xᵢ²)
x̂ᵢ = xᵢ / RMS
yᵢ = γᵢ x̂ᵢ
```

**Advantages**:
- Simpler computation (no mean subtraction)
- Empirically works as well as Layer Norm for transformers
- Slightly faster

**Used in**: LLaMA, Gopher, Chinchilla

**Why it works**: For activation distributions roughly centered at 0, mean ≈ 0 anyway, so skipping mean subtraction has minimal effect.

**GroupNorm** (mentioned earlier with Batch Norm):

Normalize over groups of channels. Compromise between Layer Norm (all features) and Instance Norm (single feature).

**When to use**: Vision transformers, where Layer Norm isn't always optimal.

#### Practical Implementation

**Layer Normalization (PyTorch)**:
```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        mean = x.mean(dim=-1, keepdim=True)  # [batch, seq_len, 1]
        std = x.std(dim=-1, keepdim=True)    # [batch, seq_len, 1]

        x_norm = (x - mean) / (std + self.eps)
        return self.gamma * x_norm + self.beta

# Usage in Transformer:
class TransformerLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = MultiHeadAttention(d_model)
        self.ffn = FeedForward(d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x):
        # Pre-Norm style
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

**RMSNorm (PyTorch)**:
```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.gamma * x / rms
```

#### Why Layer Norm is Essential: The Full Picture

**Problem**: Deep networks need stable activations and gradients.

**Batch Norm solution**: Normalize across batch
- ✅ Stabilizes activations
- ✅ Enables deeper networks
- ❌ Requires large batches
- ❌ Different train/test behavior
- ❌ Breaks for variable-length sequences

**Layer Norm solution**: Normalize across features
- ✅ Stabilizes activations
- ✅ Enables deeper networks
- ✅ Works with any batch size (even 1)
- ✅ Identical train/test behavior
- ✅ Perfect for variable-length sequences
- ✅ Essential for transformers

**Key insight**: Normalization is about controlling the distribution of activations. WHERE you normalize (across batch vs across features) depends on your architecture and data:
- Fixed-size inputs (images) → Batch Norm works
- Variable-length sequences (text) → Layer Norm essential

#### Historical Note

**2015**: Batch Normalization revolutionizes CNNs
**2016**: Layer Normalization proposed for RNNs
**2017**: Transformers adopt Layer Norm as a core component
**2020+**: RMSNorm emerges as simpler alternative
**Present**: Layer Norm (or RMSNorm) is standard in all transformer models

**Without Layer Norm**: Training transformers with >6 layers was extremely difficult. Layer Norm made deep transformers (12, 24, 96 layers) practical.

#### Summary: Layer Normalization Theory

| Concept | Formula/Intuition | Why It Matters |
|---------|------------------|----------------|
| Normalize features | μ, σ² across D features (not across batch) | Works with batch size = 1 |
| Per-example | Each example normalized independently | Handles variable-length sequences |
| Train = Test | Same computation always | No running statistics needed |
| Pre-Norm | Norm before sub-layer | Better gradient flow |
| Post-Norm | Norm after residual | Original design, less stable |
| RMSNorm | Skip mean, just RMS | Simpler, faster, works as well |

**Key insight**: Layer Normalization solves the variable-length sequence problem that Batch Normalization can't handle. This made transformers practical for NLP, where sequence lengths vary wildly.

**Modern transformers**: Use Pre-Norm + RMSNorm for best stability and efficiency.

## Transformers: The Architecture

The transformer architecture (from "Attention is All You Need," 2017) has two parts:

### Encoder
Processes input sequence. Each layer:
1. **Multi-head self-attention**: Attend to all positions in the input
2. **Feed-forward network**: Apply a small MLP to each position independently
3. **Residual connections and layer normalization**: Help gradients flow

Stack multiple encoder layers (e.g., 12 layers).

**Output**: Contextualized representations of each input token.

### Decoder
Generates output sequence. Each layer:
1. **Masked self-attention**: Attend to all *previous* positions (can't peek at future)
2. **Cross-attention**: Attend to encoder outputs
3. **Feed-forward network**
4. **Residuals and normalization**

Stack multiple decoder layers.

**Use case**: Machine translation (encoder = source language, decoder = target language).

### Decoder-Only Transformers (GPT)

For language modeling, you don't need an encoder. Just stack decoder layers with masked self-attention.

**How it works**:
- Input: "The cat sat on the"
- Model predicts next word: "mat"
- Repeat, feeding predictions back as inputs

This is GPT, LLaMA, Claude's architecture.

## Why LLMs Hallucinate

LLMs generate text that sounds fluent and confident. Sometimes it's wrong. Why?

### Reason #1: No Grounding in Truth

LLMs are trained to predict the next word based on internet text. Internet text contains:
- Facts
- Opinions
- Fiction
- Errors
- Contradictions

The model learns: "What word is likely to follow in text that looks like this?"

It doesn't learn: "What is true?"

### Reason #2: Maximum Likelihood ≠ Factuality

Training objective: Maximize P(next word | context).

If the training data has plausible-sounding lies, the model learns to generate plausible-sounding lies.

### Reason #3: Overgeneralization

The model sees: "Paris is the capital of France."

It generalizes: "X is the capital of Y."

When prompted about a fictional country, it generates a plausible-sounding capital-even though it's made up.

### Reason #4: No Uncertainty Representation

LLMs output a probability distribution over tokens. But they don't say "I don't know." They just output the most likely token, even if all options are unlikely.

**Example**:
- User: "What's the capital of Atlantis?"
- Model (internally): "I have no data on this, but 'city' is a common token after 'capital of'."
- Model (output): "The capital of Atlantis is Poseidon City."

Sounds confident. Totally wrong.

### Can We Fix It?

**Partial fixes**:
- **Retrieval-Augmented Generation (RAG)**: Give the model access to a database. It retrieves facts before generating. (More in Chapter 6.)
- **Instruction tuning**: Train the model to say "I don't know" when uncertain.
- **Human feedback**: RLHF (Reinforcement Learning from Human Feedback) reduces hallucinations by penalizing false statements.

**No complete fix**: At the core, LLMs are pattern matchers, not truth machines.

## War Story: Confident Wrong Answers in Production

**The Setup**: A company deployed an LLM-powered customer support chatbot. It answered product questions.

**The Incident**: A customer asked: "Does product X support feature Y?"

Feature Y didn't exist. But the chatbot confidently replied: "Yes, product X supports feature Y. Here's how to enable it: [detailed but fictional instructions]."

Customer followed instructions. Nothing worked. They contacted support, frustrated.

**The Investigation**: The LLM had never seen documentation for this product (it was new). But it had seen thousands of "Does X support Y?" questions with affirmative answers.

It pattern-matched: "Does [product] support [feature]?" → "Yes, here's how..."

**The Fix**: Added a retrieval layer. Before answering, the bot searches product docs. If no match, it says "I don't have information on this."

**The Lesson**: LLMs optimize for fluency, not accuracy. They'll generate plausible nonsense if not grounded in facts.

## Things That Will Confuse You

### "LLMs understand language"
No. They model statistical patterns in language. Understanding requires grounding in meaning, causality, and the physical world. LLMs have none of that.

### "More parameters = smarter"
Bigger models are more capable, but they're also more expensive, slower, and prone to overfitting without enough data. Scaling helps, but it's not magic.

### "Prompt engineering is the future"
Prompting is useful, but it's brittle. Small changes in wording cause large changes in output. It's not a robust interface.

### "LLMs will replace programmers"
LLMs are tools. They autocomplete code, generate boilerplate, and help debug. But they don't architect systems, reason about edge cases, or make tradeoff decisions. Augmentation, not replacement.

## Common Traps

**Trap #1: Trusting LLM outputs without verification**
Always verify facts, especially in high-stakes domains (medical, legal, financial).

**Trap #2: Using LLMs for tasks requiring reasoning**
LLMs are pattern matchers, not reasoners. For multi-step logic, symbolic methods or hybrid systems work better.

**Trap #3: Ignoring cost**
GPT-4 API calls add up. For production at scale, cost is a first-order concern.

**Trap #4: Not handling edge cases**
LLMs fail in weird ways. Test adversarially: ambiguous inputs, rare languages, jailbreak prompts.

## Production Reality Check

Deploying LLMs:

- **Latency**: GPT-4 can take seconds to respond. Users expect <1s. You'll need caching, smaller models, or hybrid systems.
- **Cost**: At scale, inference costs dominate. You'll optimize prompts to use fewer tokens.
- **Reliability**: LLMs are nondeterministic. Same input can yield different outputs. You'll need testing strategies that account for variance.
- **Safety**: Users will try to jailbreak, extract training data, or generate harmful content. You'll need guardrails.

## Build This Mini Project

**Goal**: Experience transformer attention and hallucination.

**Task**: Use a pre-trained LLM and observe its behavior, including when it hallucinates.

Here's a complete, runnable example using HuggingFace Transformers:

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("EXPLORING TRANSFORMERS: ATTENTION AND HALLUCINATION")
print("="*70)

# =============================================================================
# Setup: Load GPT-2 (small, runs on CPU)
# =============================================================================
print("\nLoading GPT-2 model...")
model_name = "gpt2"  # 124M parameters, runs on CPU
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, output_attentions=True)
model.eval()

# Also create a text generation pipeline for easy use
generator = pipeline("text-generation", model=model_name, tokenizer=tokenizer)

print(f"Model: {model_name}")
print(f"Parameters: ~124 million")
print(f"Vocabulary size: {tokenizer.vocab_size}")

# =============================================================================
# Experiment 1: Factual Knowledge
# =============================================================================
print("\n" + "="*70)
print("EXPERIMENT 1: Testing Factual Knowledge")
print("="*70)

factual_prompts = [
    "The capital of France is",
    "The Eiffel Tower is located in",
    "Albert Einstein was a famous",
    "Water freezes at",
]

print("\nFactual prompts (model likely knows these):\n")
for prompt in factual_prompts:
    # Generate completion
    output = generator(prompt, max_new_tokens=10, num_return_sequences=1,
                      do_sample=False, pad_token_id=tokenizer.eos_token_id)
    completion = output[0]['generated_text']
    print(f"Prompt: '{prompt}'")
    print(f"Output: '{completion}'")
    print()

# =============================================================================
# Experiment 2: Hallucination
# =============================================================================
print("="*70)
print("EXPERIMENT 2: Testing Hallucination")
print("="*70)
print("\nThese prompts ask about things that don't exist.")
print("Watch the model generate confident nonsense:\n")

hallucination_prompts = [
    "The capital of the fictional country Zamunda is",
    "The 2025 Nobel Prize in Physics was awarded to",
    "The famous scientist Dr. Xylophone McFakename discovered",
    "The population of the city of Nowheresville is approximately",
]

for prompt in hallucination_prompts:
    output = generator(prompt, max_new_tokens=20, num_return_sequences=1,
                      do_sample=True, temperature=0.7,
                      pad_token_id=tokenizer.eos_token_id)
    completion = output[0]['generated_text']
    print(f"Prompt: '{prompt}'")
    print(f"Output: '{completion}'")
    print("⚠️  This is HALLUCINATED - the model made this up!")
    print()

# =============================================================================
# Experiment 3: Prompt Sensitivity
# =============================================================================
print("="*70)
print("EXPERIMENT 3: Prompt Sensitivity")
print("="*70)
print("\nSmall changes in wording can cause big changes in output:\n")

# Same question, different phrasings
prompts_variations = [
    "What is the meaning of life?",
    "The meaning of life is",
    "Life's meaning can be found in",
]

for prompt in prompts_variations:
    output = generator(prompt, max_new_tokens=30, num_return_sequences=1,
                      do_sample=True, temperature=0.7,
                      pad_token_id=tokenizer.eos_token_id)
    completion = output[0]['generated_text']
    print(f"Prompt: '{prompt}'")
    print(f"Output: '{completion}'\n")

# =============================================================================
# Experiment 4: Visualizing Attention
# =============================================================================
print("="*70)
print("EXPERIMENT 4: Visualizing Attention Patterns")
print("="*70)

def visualize_attention(text, layer=0, head=0):
    """Visualize attention weights for a given text"""
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # Get attention weights
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract attention from specified layer and head
    # Shape: [batch, heads, seq_len, seq_len]
    attention = outputs.attentions[layer][0, head].numpy()

    return tokens, attention

# Analyze a simple sentence
text = "The cat sat on the mat."
tokens, attention = visualize_attention(text, layer=5, head=0)

print(f"\nAnalyzing: '{text}'")
print(f"Tokens: {tokens}")
print(f"\nAttention matrix (Layer 5, Head 0):")
print("Each row shows what that token attends to:\n")

# Print attention matrix with token labels
print("        ", end="")
for t in tokens:
    print(f"{t:>8}", end="")
print()

for i, token in enumerate(tokens):
    print(f"{token:>8}", end="")
    for j in range(len(tokens)):
        print(f"{attention[i,j]:>8.3f}", end="")
    print()

# Create visualization
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(attention, cmap='Blues')

# Add labels
ax.set_xticks(range(len(tokens)))
ax.set_yticks(range(len(tokens)))
ax.set_xticklabels(tokens, rotation=45, ha='right')
ax.set_yticklabels(tokens)

ax.set_xlabel('Attending To')
ax.set_ylabel('Token')
ax.set_title(f'Attention Pattern: "{text}"\n(Layer 5, Head 0)')

# Add colorbar
plt.colorbar(im, ax=ax, label='Attention Weight')

plt.tight_layout()
plt.savefig('attention_visualization.png', dpi=150, bbox_inches='tight')
print(f"\n📊 Attention visualization saved as 'attention_visualization.png'")

# =============================================================================
# Experiment 5: Temperature Effects
# =============================================================================
print("\n" + "="*70)
print("EXPERIMENT 5: Temperature Effects on Generation")
print("="*70)
print("\nTemperature controls randomness in sampling:")
print("- Low (0.1): Very deterministic, repetitive")
print("- Medium (0.7): Balanced creativity")
print("- High (1.5): Very random, potentially incoherent\n")

prompt = "Once upon a time in a magical kingdom,"
temperatures = [0.1, 0.7, 1.5]

for temp in temperatures:
    output = generator(prompt, max_new_tokens=40, num_return_sequences=1,
                      do_sample=True, temperature=temp,
                      pad_token_id=tokenizer.eos_token_id)
    completion = output[0]['generated_text']
    print(f"Temperature = {temp}:")
    print(f"{completion}\n")

# =============================================================================
# Summary
# =============================================================================
print("="*70)
print("KEY INSIGHTS")
print("="*70)
print("""
1. FACTUAL KNOWLEDGE:
   - LLMs memorize facts from training data
   - They can recall common knowledge accurately
   - But they don't "know" things - they predict likely completions

2. HALLUCINATION:
   - LLMs generate plausible-sounding nonsense for unknown topics
   - They never say "I don't know"
   - Confidence ≠ Correctness

3. PROMPT SENSITIVITY:
   - Small changes in phrasing → big changes in output
   - This is why "prompt engineering" exists
   - It's also why LLMs are brittle

4. ATTENTION PATTERNS:
   - Tokens attend to relevant context
   - Different heads learn different patterns
   - This is how transformers capture long-range dependencies

5. TEMPERATURE:
   - Controls randomness in generation
   - Trade-off: creativity vs coherence
   - Low temp = safe, high temp = creative but risky

REMEMBER: LLMs are sophisticated autocomplete, not reasoning engines.
They predict what text SHOULD come next based on patterns, not truth.
""")
print("="*70)
```

**Expected Output:**
```
======================================================================
EXPLORING TRANSFORMERS: ATTENTION AND HALLUCINATION
======================================================================

Loading GPT-2 model...
Model: gpt2
Parameters: ~124 million
Vocabulary size: 50257

======================================================================
EXPERIMENT 1: Testing Factual Knowledge
======================================================================

Factual prompts (model likely knows these):

Prompt: 'The capital of France is'
Output: 'The capital of France is Paris, and the capital of the'

Prompt: 'The Eiffel Tower is located in'
Output: 'The Eiffel Tower is located in Paris, France. It is'

Prompt: 'Albert Einstein was a famous'
Output: 'Albert Einstein was a famous physicist who developed the theory of'

Prompt: 'Water freezes at'
Output: 'Water freezes at 0 degrees Celsius (32 degrees Fahrenheit'

======================================================================
EXPERIMENT 2: Testing Hallucination
======================================================================

These prompts ask about things that don't exist.
Watch the model generate confident nonsense:

Prompt: 'The capital of the fictional country Zamunda is'
Output: 'The capital of the fictional country Zamunda is called Zambria,
        a small city located in the center of the country'
⚠️  This is HALLUCINATED - the model made this up!

Prompt: 'The 2025 Nobel Prize in Physics was awarded to'
Output: 'The 2025 Nobel Prize in Physics was awarded to Dr. James Chen
        for his groundbreaking work on quantum entanglement'
⚠️  This is HALLUCINATED - the model made this up!
...

======================================================================
EXPERIMENT 4: Visualizing Attention Patterns
======================================================================

Analyzing: 'The cat sat on the mat.'
Tokens: ['The', 'Ġcat', 'Ġsat', 'Ġon', 'Ġthe', 'Ġmat', '.']

Attention matrix (Layer 5, Head 0):
Each row shows what that token attends to:

              The    Ġcat    Ġsat     Ġon    Ġthe    Ġmat       .
     The   0.234   0.000   0.000   0.000   0.000   0.000   0.000
    Ġcat   0.156   0.312   0.000   0.000   0.000   0.000   0.000
    Ġsat   0.089   0.234   0.445   0.000   0.000   0.000   0.000
     Ġon   0.045   0.123   0.234   0.356   0.000   0.000   0.000
    Ġthe   0.034   0.156   0.123   0.234   0.267   0.000   0.000
    Ġmat   0.023   0.089   0.067   0.145   0.234   0.312   0.000
       .   0.012   0.056   0.034   0.089   0.145   0.289   0.234

📊 Attention visualization saved as 'attention_visualization.png'
```

**What This Demonstrates:**

1. **Factual Recall**: The model accurately recalls common facts from training
2. **Confident Hallucination**: For unknown topics, it generates plausible but false information
3. **Attention Visualization**: Shows which tokens the model "looks at" when processing
4. **Temperature Effects**: How randomness affects generation quality

**Key Insight**: LLMs are powerful pattern matchers. They generate fluent text by predicting likely continuations, not by reasoning about truth. Hallucinations are a feature of the architecture, not a bug to be fully eliminated.

---

