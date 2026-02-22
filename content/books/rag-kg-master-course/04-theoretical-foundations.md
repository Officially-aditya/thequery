# 2A. THEORETICAL FOUNDATIONS (DEEP DIVE)

> **"Theory without practice is sterile, practice without theory is blind."** - Immanuel Kant

This section provides the mathematical and conceptual foundations that power RAG and Knowledge Graph systems. Understanding these principles deeply will transform you from a code copier to an AI systems architect.

**Fair warning**: This section is dense. We're going to cover vector space theory, information retrieval theory, graph theory, and semantic similarity - all with actual math. If you're here to copy-paste code and move on, skip this section and come back when something breaks in production and you need to understand why. If you want to be the person who designs the system rather than just using it, buckle up.

---

## Vector Space Theory & Embeddings (The Mathematics of Meaning)

### The Core Idea: Meaning as Geometry

**Fundamental Insight**: If we can represent words, sentences, or documents as points in a high-dimensional space, then similar meanings should be close together geometrically.

**Historical Context**: This idea dates back to distributional semantics (1950s): *"You shall know a word by the company it keeps"* - J.R. Firth. Modern embeddings (Word2Vec 2013, BERT 2018) are the mathematical realization of this principle.

### Vector Spaces: A Primer

A **vector space** is a mathematical structure where:
- Each point is represented by coordinates: **v** = [v₁, v₂, ..., vₙ]
- You can add vectors and multiply by scalars
- Distance and angle have meaning

**Example in 2D**:
```
Word "king" = [0.5, 0.8]
Word "queen" = [0.4, 0.7]
Word "man" = [0.3, 0.1]
Word "woman" = [0.2, 0.0]

Geometry shows: king - man + woman ≈ queen
```

This is the famous word analogy property!

### Why High Dimensions?

**Real embeddings use 768-4096 dimensions**. Why so many?

(This is one of those questions that seems simple but has a deep answer. The short version: we need enough dimensions to keep millions of different meanings separate. The long version involves manifold theory and will make your head hurt. We'll give you both.)

**Curse of Dimensionality Paradox**: In high dimensions:
- Most points are far from each other (good for distinguishing meanings)
- But angles become more meaningful than distances
- More capacity to encode subtle semantic distinctions

**Information Content**: Language has ~10⁵ common words × multiple senses = need high dimensions to keep them separated.

#### The Mathematical Justification for High Dimensionality

The necessity of high-dimensional embeddings can be rigorously understood through the **Johnson-Lindenstrauss Lemma**, which states that a set of points in high-dimensional space can be embedded into a lower-dimensional space while approximately preserving pairwise distances.

**Formal Statement**: For any 0 < ε < 1, a set of n points in ℝ^D can be embedded into ℝ^k where k = O(log(n)/ε²), such that all pairwise distances are preserved within a factor of (1 ± ε).

**Implications for Embeddings**:
- With 100,000 words and ε = 0.1 (10% error tolerance), we need k ≈ 115,000 dimensions theoretically
- However, semantic structure has redundancy and lower intrinsic dimensionality
- Modern embeddings (768-1536 dims) represent a practical compromise between:
  - **Expressiveness**: Enough dimensions to separate distinct meanings
  - **Computational efficiency**: Small enough for fast similarity computation
  - **Statistical efficiency**: Not so high that we need enormous training data

#### Intrinsic Dimensionality of Language

Research suggests that while embeddings use 768+ dimensions, the **intrinsic dimensionality** of semantic space is much lower (estimated 50-200 dimensions). This means:

1. **Manifold Hypothesis**: Semantic meanings lie on a lower-dimensional manifold embedded in high-dimensional space
2. **Redundancy**: Many dimensions encode similar information (entangled representations)
3. **Optimization**: High dimensions make training easier (less local minima) even if not all are strictly necessary

**Empirical Evidence**:
```python
# Principal Component Analysis on word2vec embeddings
# Typically shows that 95% of variance captured in ~100 principal components
# Yet we use 300 dims because:
# - Easier to train
# - Better generalization
# - Captures rare semantic distinctions
```

#### The Geometry of Meaning: A Deeper Dive

**Vector Space Axioms Applied to Semantics**:

A vector space V over a field F (typically ℝ for embeddings) satisfies:

1. **Closure under addition**: v + w ∈ V for all v, w ∈ V
   - Semantic meaning: Combining concepts creates new concepts
   - Example: "king" + "crown" = "monarchy"

2. **Associativity**: (u + v) + w = u + (v + w)
   - Meaning composition is consistent regardless of grouping

3. **Existence of zero vector**: ∃ 0 ∈ V such that v + 0 = v
   - The "null meaning" or "no information" vector

4. **Existence of additive inverse**: For each v, ∃ -v such that v + (-v) = 0
   - Semantic opposites: "hot" + "cold" ≈ neutral

5. **Scalar multiplication**: α·v ∈ V for all α ∈ ℝ, v ∈ V
   - Intensity or magnitude of meaning
   - Example: 2·"happy" = "very happy", 0.5·"run" = "jog"

**Why This Mathematical Structure Matters**:

The vector space structure enables **algebraic reasoning about meaning**:
- We can "solve" for unknown concepts: "king" - "man" + "woman" = ?
- We can interpolate: 0.7·"walk" + 0.3·"run" ≈ "jog"
- We can find orthogonal (unrelated) concepts using the nullspace

### The Mathematics of Embeddings

#### 1. Cosine Similarity (The Core Metric)

**Why cosine, not Euclidean distance?**

(This question comes up constantly. "Why not just use normal distance?" Because normal distance is fooled by vector magnitude. A 10-page essay and a 1-sentence summary might have identical meaning but very different vector magnitudes. Cosine similarity only cares about direction, not length. This makes it scale-invariant, which is exactly what you want for semantic similarity.)

Given two vectors **u** and **v**:

```
Euclidean Distance: d(u,v) = ||u - v|| = √(Σ(uᵢ - vᵢ)²)
Cosine Similarity:  cos(θ) = (u·v)/(||u|| ||v||) = Σ(uᵢvᵢ)/√(Σuᵢ²)√(Σvᵢ²)
```

**Cosine similarity ranges from -1 to 1**:
- 1 = same direction (identical meaning)
- 0 = orthogonal (unrelated)
- -1 = opposite direction (antonyms)

**Why cosine wins**:
- **Scale invariant**: "good movie" and "really really good movie" should be similar despite different vector magnitudes
- **Normalized**: Always in [-1, 1], easy to interpret
- **Angular**: Captures semantic relationship independent of frequency

**Geometric Intuition**:
```
         u  /
           /θ
          /_____ v

cos(θ) = how aligned the vectors are
Small θ → cos(θ) ≈ 1 → very similar
Large θ → cos(θ) ≈ 0 → unrelated
```

#### Mathematical Properties of Cosine Similarity

**1. Relationship to Dot Product**:

For normalized vectors (||u|| = ||v|| = 1), cosine similarity reduces to the dot product:
```
cos(θ) = u · v = Σᵢ uᵢvᵢ
```

This is why many vector databases normalize embeddings and use dot product for speed!

**2. Metric Properties (or lack thereof)**:

Cosine similarity is NOT a metric because it violates the triangle inequality. However, we can convert it to a metric:

```
d_cos(u,v) = 1 - cos(u,v)  (cosine distance)
```

Or for a proper metric:
```
d_angular(u,v) = arccos(cos(u,v)) / π
```

This gives values in [0,1] and satisfies triangle inequality.

**3. Computational Optimization**:

For large-scale retrieval:
```python
# Naive: O(nd) for n documents, d dimensions
similarities = [cosine(query, doc) for doc in documents]

# Optimized with normalization + matrix multiplication: O(nd) but much faster
# Normalize once
docs_normalized = docs / np.linalg.norm(docs, axis=1, keepdims=True)
query_normalized = query / np.linalg.norm(query)

# Single matrix multiplication
similarities = np.dot(docs_normalized, query_normalized)
```

**4. Why Cosine for Text?**:

Theoretical justification from **distributional semantics**:

- Document vectors represent word co-occurrence statistics
- Longer documents have larger magnitude but same semantic content
- Cosine normalizes away document length, focusing on **word distribution**

**Example**:
```
Doc 1: "dog cat dog cat dog cat" → [3, 3]
Doc 2: "dog cat"                  → [1, 1]

Euclidean distance: ||[3,3] - [1,1]|| = 2.83 (seems different!)
Cosine similarity:  [3,3]·[1,1] / (|[3,3]||[1,1]|) = 6/(4.24*1.41) = 1.0 (identical!)
```

Both documents have the same semantic content (50% dog, 50% cat), and cosine correctly identifies this.

#### Alternative Similarity Measures and When to Use Them

**Euclidean Distance (L2)**:
```
d(u,v) = ||u - v|| = √(Σ(uᵢ - vᵢ)²)
```
- **Use when**: Magnitude matters (e.g., embedding dense entities with varying importance)
- **RAG application**: Less common, but useful for hierarchical embeddings

**Manhattan Distance (L1)**:
```
d(u,v) = Σ|uᵢ - vᵢ|
```
- **Use when**: Sparse vectors, interpretable dimensions
- **RAG application**: TF-IDF vectors, bag-of-words

**Dot Product (Inner Product)**:
```
u · v = Σ uᵢvᵢ
```
- **Use when**: Vectors are normalized OR magnitude encodes importance
- **RAG application**: Fast approximate nearest neighbor search (FAISS uses this)

**Comparison Table**:
```
Similarity Measure | Normalized? | Metric? | Speed    | Best For
-------------------|-------------|---------|----------|------------------
Cosine             | Yes         | No*     | Fast     | Text embeddings
Euclidean          | No          | Yes     | Fast     | Dense vectors
Dot Product        | No          | No      | Fastest  | Pre-normalized
Manhattan          | No          | Yes     | Fast     | Sparse vectors

*Cosine distance (1-cos) forms a pseudo-metric
```

#### 2. How Embeddings Are Learned

**Skip-gram (Word2Vec) - The Original Insight**:

**Objective**: Predict context words from center word

Given sentence: "The quick brown fox jumps"
- Center: "brown"
- Context: ["quick", "fox"]

**Neural Network**:
```
Input: one-hot vector for "brown" [0,0,1,0,0,...]
       ↓
Hidden Layer (embedding): [0.2, -0.5, 0.8, ...] ← This is the embedding!
       ↓
Output: probability distribution over context words
```

**Training**: Adjust embeddings so that words appearing in similar contexts get similar vectors.

**Mathematical Formulation**:

Maximize: Σ log P(context | word)

Where: P(context | word) = exp(u_context · v_word) / Σ exp(u_i · v_word)

This is **softmax** - converts dot products into probabilities.

**Detailed Training Objective**:

For a corpus with vocabulary V and sequence of words w₁, w₂, ..., wₜ, the skip-gram objective is:

```
J(θ) = (1/T) Σₜ Σ₋c≤j≤c,j≠0 log P(wₜ₊ⱼ | wₜ)

where:
- T = total words in corpus
- c = context window size (typically 5-10)
- wₜ = target word at position t
- wₜ₊ⱼ = context word at offset j

P(wₒ | wᵢ) = exp(u_o^T v_i) / Σₖ₌₁^V exp(u_k^T v_i)
```

**The Computational Challenge**:

The softmax denominator requires summing over entire vocabulary V (typically 100k+ words), making this O(V) per training example.

**Solution: Negative Sampling**

Instead of computing full softmax, approximate with:
```
log σ(u_o^T v_i) + Σₖ₌₁^K 𝔼_k~P_n[log σ(-u_k^T v_i)]

where:
- σ(x) = 1/(1+e^-x) is sigmoid function
- K = number of negative samples (typically 5-20)
- P_n = noise distribution (usually unigram^(3/4))
```

**Why This Works**:
- True context word: Maximize σ(u_o^T v_i) → push u_o and v_i closer
- Random negative samples: Maximize σ(-u_k^T v_i) → push u_k and v_i apart
- Complexity: O(K) instead of O(V) - huge speedup!

**The Two Embedding Matrices**:

Word2Vec actually learns TWO embeddings per word:
1. **v_w**: Word as center (when predicting context)
2. **u_w**: Word as context (when being predicted)

Final embedding typically averages or concatenates these, giving richer representation.

#### Skip-gram vs. CBOW (Continuous Bag of Words)

**CBOW**: Inverse of skip-gram - predict center word from context

```
Skip-gram:  center → context words
CBOW:       context words → center

Training Signal:
Skip-gram:  "The quick [brown] fox jumps" → predict "quick", "fox"
CBOW:       "The quick [?] fox jumps" → predict "brown" from context
```

**When to use each**:
- **Skip-gram**: Better for small datasets, rare words, captures more nuanced semantics
- **CBOW**: Faster training, better for frequent words, smoother embeddings

**Mathematical Difference**:
```
Skip-gram: P(context | center) = Πᵢ P(wᵢ | w_center)
CBOW:      P(center | context) = P(w_center | avg(context_words))
```

#### GloVe (Global Vectors) - The Matrix Factorization View

**Key Insight**: Embeddings can also be learned by factorizing word co-occurrence matrices.

**Objective**:
```
J = Σᵢ,ⱼ f(Xᵢⱼ)(wᵢ^T w̃ⱼ + bᵢ + b̃ⱼ - log Xᵢⱼ)²

where:
- Xᵢⱼ = number of times word j appears in context of word i
- wᵢ, w̃ⱼ = word embeddings
- bᵢ, b̃ⱼ = bias terms
- f(x) = weighting function (gives less weight to very frequent/rare pairs)
```

**Weighting Function**:
```
f(x) = (x/x_max)^α if x < x_max, else 1
```

This prevents overfitting to very frequent co-occurrences like "the the".

**Why GloVe Matters**:
- Bridges word2vec (local context) and LSA (global statistics)
- Often produces slightly better analogies than word2vec
- More interpretable (directly models co-occurrence)

**Transformer Embeddings (BERT, GPT) - The Modern Approach**:

Instead of static embeddings, transformers create **contextualized embeddings**:

"I went to the **bank** to deposit money" → bank₁ = [0.1, 0.8, ...]
"I sat by the river **bank**" → bank₂ = [0.5, 0.2, ...]

**How?** Attention mechanism (we'll cover next) that looks at surrounding words.

#### The Attention Mechanism (Intuition)

**Problem with Static Embeddings**: "bank" always gets same vector regardless of context.

**Solution**: Compute embedding as weighted combination of all words in sentence.

**Simplified Attention Formula**:
```
For word wᵢ in sentence w₁, w₂, ..., wₙ:

1. Compute attention scores: αᵢⱼ = score(wᵢ, wⱼ)
2. Normalize: âᵢⱼ = softmax(αᵢⱼ) = exp(αᵢⱼ) / Σₖ exp(αᵢₖ)
3. Contextualized embedding: hᵢ = Σⱼ âᵢⱼ · v_wⱼ

where v_wⱼ is the initial embedding of word j
```

**Example**:
```
Sentence: "The bank by the river"

For "bank":
- High attention to: "river" (0.4), "the" (before bank, 0.3)
- Low attention to: "The" (start, 0.1), "by" (0.1), "the" (after, 0.1)

Final embedding: 0.4·v_river + 0.3·v_the + 0.1·v_The + 0.1·v_by + 0.1·v_the
→ Heavily influenced by "river", encodes "financial institution" less
```

**Self-Attention in Transformers**:

The actual mechanism is more sophisticated (Query-Key-Value):

```
For each word wᵢ:
- Query: Qᵢ = wᵢ · W_Q  (what am I looking for?)
- Key:   Kⱼ = wⱼ · W_K  (what do I contain?)
- Value: Vⱼ = wⱼ · W_V  (what do I communicate?)

Attention(Q,K,V) = softmax(QK^T / √d_k) V

where:
- d_k = dimension of key vectors
- Division by √d_k prevents vanishing gradients
```

**Why This Works**:
- Different words attend to different contexts
- Multi-head attention captures different relationships (syntax, semantics, etc.)
- Stacked layers build increasingly abstract representations

#### 3. Properties of Good Embeddings

**Linearity**: Semantic relationships are linear transformations
```
king - man + woman ≈ queen
Paris - France + Italy ≈ Rome
```

**Clustering**: Similar concepts cluster together
```
Fruits: [apple, orange, banana] are close in space
Animals: [dog, cat, lion] form another cluster
```

**Dimensionality**: Each dimension captures a semantic feature
- Dimension 42 might encode "royalty"
- Dimension 108 might encode "gender"
- (Though in practice, dimensions are entangled)

### Embedding Quality Metrics

#### Intrinsic Evaluation:

**1. Word Similarity**:
Correlation with human judgments (WordSim-353 dataset)

**2. Analogy Accuracy**:
"man:woman :: king:?" → should predict "queen"

#### Extrinsic Evaluation:

**Downstream Task Performance**: How well does retrieval work?

### Practical Implications for RAG

**Why this matters for your RAG system**:

1. **Chunk Size**: Larger chunks → more diverse content → lower quality embeddings
   - Sweet spot: 200-1000 tokens per chunk
   - Each chunk should have coherent semantic content

2. **Query Expansion**: If query is short, expand it before embedding
   - Short query: "RAG" → poor embedding
   - Expanded: "What is retrieval-augmented generation?" → better

3. **Embedding Model Choice**:
   - General models (OpenAI): Good for diverse content
   - Domain-specific: Train on your corpus for 10-20% improvement

4. **Similarity Threshold**: Not all cosine scores are created equal
   - 0.9+ : Very similar (same topic, same phrasing)
   - 0.7-0.9 : Similar (same topic, different phrasing)
   - 0.5-0.7 : Related (adjacent topics)
   - <0.5 : Probably not relevant

---

## Information Retrieval Theory (The Science of Finding)

### What is Information Retrieval?

**Formal Definition**: Given a query Q and document collection D, find documents D' ⊂ D that are relevant to Q.

**Core Challenge**: "Relevance" is subjective and context-dependent.

### Classic IR: TF-IDF (Term Frequency-Inverse Document Frequency)

**The Intuition**:
- Words that appear often in a document are important for that document (TF)
- But words that appear in all documents are less discriminative (IDF)

**Mathematics**:

**Term Frequency** (how often term t appears in document d):
```
TF(t,d) = count(t,d) / |d|
```

**Inverse Document Frequency** (how rare is term t):
```
IDF(t) = log(N / df(t))

where:
- N = total documents
- df(t) = documents containing term t
```

**TF-IDF Score**:
```
TF-IDF(t,d) = TF(t,d) × IDF(t)
```

**Example**:

Document: "The cat sat on the mat"
Query: "cat"

```
TF("cat") = 1/6 = 0.167
IDF("cat") = log(1000/50) = 2.996  (if 50 docs out of 1000 mention "cat")
TF-IDF = 0.167 × 2.996 = 0.500

vs.

TF("the") = 2/6 = 0.333
IDF("the") = log(1000/999) = 0.001  ("the" is in almost every document)
TF-IDF = 0.333 × 0.001 = 0.0003
```

**Insight**: Common words get downweighted automatically!

### BM25: The King of Lexical Retrieval

**BM25** (Best Matching 25) improves on TF-IDF with **diminishing returns** and **length normalization**.

**The Formula** (don't memorize, understand the components):

```
BM25(Q,D) = Σ IDF(qᵢ) · (f(qᵢ,D) · (k₁ + 1)) / (f(qᵢ,D) + k₁ · (1 - b + b · |D|/avgdl))
            qᵢ∈Q

where:
- f(qᵢ,D) = frequency of term qᵢ in document D
- |D| = length of document D
- avgdl = average document length
- k₁ = term frequency saturation (usually 1.2-2.0)
- b = length normalization (usually 0.75)
```

**What each part does**:

1. **IDF(qᵢ)**: Rare terms are more important (like TF-IDF)

2. **Saturation**: `f/(f + k₁)` approaches 1 as f increases
   - Mentioning "cat" 100 times doesn't make doc 100× more relevant
   - Diminishing returns built in!

3. **Length Normalization**: `(1 - b + b · |D|/avgdl)`
   - Longer documents naturally have higher term frequencies
   - This penalty prevents bias toward long docs

**Visual Intuition**:
```
TF-IDF: Score grows linearly with term frequency
        |        /
Score   |      /
        |    /
        |  /
        |/___________
          Term Freq

BM25: Score saturates (diminishing returns)
        |     ____
Score   |   /
        | /
        |/___________
          Term Freq
```

**Why BM25 is still used in 2025**:

Despite neural retrieval, BM25 excels at:
- **Exact matches**: "invoice #12345"
- **Rare terms**: Technical jargon, product IDs
- **Speed**: No GPU needed
- **Interpretability**: You can see which terms matched

### Neural Retrieval: Dense Passage Retrieval (DPR)

**The Paradigm Shift**: Instead of matching words, match meanings.

**Architecture**:
```
Query: "What causes rain?"
       ↓
Query Encoder (BERT)
       ↓
Query Embedding: q = [0.2, -0.5, 0.8, ...]

Document: "Precipitation occurs when water vapor condenses..."
       ↓
Document Encoder (BERT)
       ↓
Doc Embedding: d = [0.18, -0.48, 0.82, ...]

Similarity = q · d = cosine similarity
```

**Training** (Contrastive Learning):

**Positive pairs**: (query, relevant doc)
**Negative pairs**: (query, irrelevant doc)

**Loss function**:
```
L = -log(exp(q·d⁺) / (exp(q·d⁺) + Σ exp(q·dᵢ⁻)))
```

**Translation**: Make relevant doc close, irrelevant docs far.

**Why this works**:
- BERT understands "rain" and "precipitation" are related
- Captures semantic similarity, not just word overlap
- Works across languages, paraphrases

**Limitations**:
- Computationally expensive (need GPU)
- Can miss exact matches
- Less interpretable

### Hybrid Retrieval: Best of Both Worlds

**The Insight**: BM25 and neural retrieval are complementary.

**Reciprocal Rank Fusion (RRF)**: Simple but effective

```
Given two ranked lists: BM25 results and Dense results

Score(doc) = 1/(k + rank_BM25(doc)) + 1/(k + rank_dense(doc))

where k = 60 (empirically chosen constant)
```

**Example**:
```
BM25 ranks:    [doc1, doc3, doc2, doc5]
Dense ranks:   [doc2, doc1, doc4, doc3]

RRF scores:
doc1: 1/61 + 1/62 = 0.0328
doc2: 1/63 + 1/61 = 0.0322
doc3: 1/62 + 1/64 = 0.0318
...

Final ranking: [doc1, doc2, doc3, ...]
```

**Alternative: Learned Fusion**

Train a small model to combine scores:
```
score = w₁·BM25(q,d) + w₂·Dense(q,d) + w₃·BM25(q,d)·Dense(q,d)
```

Learn w₁, w₂, w₃ from data.

### Relevance and Precision-Recall

**Fundamental Tradeoff**: You can't have perfect precision and perfect recall simultaneously.

**Definitions**:
```
Precision = Relevant Retrieved / Total Retrieved
Recall = Relevant Retrieved / Total Relevant

F1 Score = 2 · (Precision · Recall) / (Precision + Recall)
```

**Visual**:
```
        Retrieved
    ┌──────────────┐
    │   ┌──────┐   │
    │   │ TP   │   │  ← True Positives (what we want!)
    │   └──────┘   │
    │      FP      │  ← False Positives (noise)
    └──────────────┘
         FN           ← False Negatives (missed relevant docs)
```

**The Curve**:
```
Precision
    ↑
  1 |╲
    | ╲
    |  ╲___
    |      ╲___
  0 |__________╲___→ Recall
    0              1

As you retrieve more docs (↑ recall), precision drops
```

**Practical Implications for RAG**:

- **Top-k = 5**: High precision, might miss relevant info
- **Top-k = 50**: Better recall, but more noise for LLM
- **Sweet spot**: 10-20 documents for most use cases

**Retrieval @ k**: Metric for RAG systems
```
Recall@5 = "What fraction of relevant docs are in top 5?"
```

---

## Graph Theory Fundamentals (The Mathematics of Relationships)

### What is a Graph? (Formally)

**Definition**: A graph G = (V, E) consists of:
- **V**: Set of vertices (nodes)
- **E**: Set of edges (relationships)

**Types of Graphs**:

1. **Directed Graph** (Digraph):
   - Edges have direction: A → B ≠ B → A
   - Example: "Alice follows Bob" (Twitter)

2. **Undirected Graph**:
   - Edges are bidirectional: A — B = B — A
   - Example: "Alice is friends with Bob" (Facebook)

3. **Weighted Graph**:
   - Edges have weights: A -(5)→ B
   - Example: Road network (weights = distance)

4. **Property Graph** (Knowledge Graphs):
   - Nodes and edges have properties
   - Example: (Person {name:"Alice"}) -[KNOWS {since:2020}]→ (Person {name:"Bob"})

### Graph Representation

#### Adjacency Matrix

For graph with n nodes:
```
     A  B  C  D
A  [ 0  1  1  0 ]
B  [ 0  0  1  1 ]
C  [ 0  0  0  1 ]
D  [ 0  0  0  0 ]

1 = edge exists, 0 = no edge
```

**Space**: O(n²)
**Edge lookup**: O(1)
**Best for**: Dense graphs (many edges)

#### Adjacency List

```
A → [B, C]
B → [C, D]
C → [D]
D → []
```

**Space**: O(n + e) where e = number of edges
**Edge lookup**: O(degree)
**Best for**: Sparse graphs (few edges) ← Most real graphs!

(If you're implementing a knowledge graph from scratch and considering an adjacency matrix because "O(1) lookup is faster," stop. Most real graphs are sparse - a typical person knows hundreds of people, not millions. An adjacency matrix for 1M nodes takes 1TB of RAM just to store zeros. Use adjacency lists. This is one of those textbook-vs-reality moments where the "slower" algorithm is actually faster in practice.)

### Graph Properties

#### Degree

**In-degree**: Number of incoming edges
**Out-degree**: Number of outgoing edges

**Example**: Twitter
- High in-degree = celebrity (many followers)
- High out-degree = active user (follows many)

#### Path

**Path**: Sequence of vertices connected by edges
- A → B → C is a path of length 2

**Shortest Path**: Minimum number of edges between two nodes
- Dijkstra's algorithm (weighted)
- BFS (unweighted)

**Why this matters for KG**:
- "How is Alice connected to Machine Learning?"
- Find shortest path: Alice → WORKS_ON → Project X → REQUIRES → Machine Learning

#### Connectedness

**Connected Graph**: Path exists between any two nodes

**Components**: Maximal connected subgraphs
```
Graph:    A─B    C─D─E
          │      │
          F      G

Components: {A,B,F}, {C,D,E,G}
```

**In KG**: Disconnected components might indicate:
- Different knowledge domains
- Data quality issues (missing links)

#### Cycles

**Cycle**: Path that starts and ends at same node
```
A → B → C → A  (cycle of length 3)
```

**DAG** (Directed Acyclic Graph): No cycles
- Used for: Ontologies, dependency graphs
- Example: File → Directory → Filesystem (no circular dependencies)

**Cyclic Graphs**: Allow cycles
- Used for: Social networks, knowledge graphs
- Example: A knows B, B knows C, C knows A

### Graph Algorithms for Knowledge Graphs

#### 1. Breadth-First Search (BFS)

**Use Case**: Find shortest path, neighborhood exploration

**Algorithm**:
```
BFS(start_node):
    queue = [start_node]
    visited = {start_node}

    while queue not empty:
        node = queue.pop()
        for neighbor in node.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

**Complexity**: O(V + E)

**KG Application**: "Find all skills within 2 hops of Alice"

#### 2. Depth-First Search (DFS)

**Use Case**: Path finding, cycle detection

**Algorithm**:
```
DFS(node, visited):
    visited.add(node)
    for neighbor in node.neighbors:
        if neighbor not in visited:
            DFS(neighbor, visited)
```

**Complexity**: O(V + E)

**KG Application**: "Find any path from Alice to Bob"

#### 3. PageRank (The Google Algorithm)

**Intuition**: A node is important if important nodes point to it.

**Mathematical Formulation**:
```
PR(A) = (1-d)/N + d · Σ PR(Tᵢ)/C(Tᵢ)
                      i

where:
- d = damping factor (usually 0.85)
- N = total nodes
- Tᵢ = nodes pointing to A
- C(Tᵢ) = out-degree of Tᵢ
```

**Iterative Computation**:
```
Initialize: PR(node) = 1/N for all nodes
Repeat until convergence:
    For each node A:
        PR_new(A) = (1-d)/N + d · Σ PR_old(Tᵢ)/C(Tᵢ)
```

**KG Application**: "Find the most influential people in the organization"

**Example**:
```
Graph:   A ← B
         ↓   ↓
         C ← D

After convergence:
PR(C) > PR(D) > PR(A) > PR(B)

C is most important (receives links from important nodes A and D)
```

#### 4. Community Detection (Louvain Algorithm)

**Goal**: Find clusters of densely connected nodes

**Metric**: Modularity
```
Q = 1/(2m) Σ [Aᵢⱼ - (kᵢkⱼ)/(2m)] δ(cᵢ,cⱼ)

where:
- m = total edges
- Aᵢⱼ = edge between i and j
- kᵢ = degree of i
- cᵢ = community of i
- δ(cᵢ,cⱼ) = 1 if i,j in same community
```

**Intuition**: More edges within communities than expected by chance

**KG Application**: "Find research groups with shared interests"

#### 5. Dijkstra's Algorithm (Shortest Path in Weighted Graphs)

**Problem**: Find shortest path in graph with positive edge weights

**Use Case**: "What's the fastest way to learn Machine Learning given prerequisites?"

**Algorithm**:
```python
import heapq

def dijkstra(graph, start, end):
    """
    Find shortest path in weighted graph
    graph: {node: [(neighbor, weight), ...]}
    """
    # Priority queue: (distance, node, path)
    pq = [(0, start, [start])]
    visited = set()
    distances = {start: 0}

    while pq:
        current_dist, current, path = heapq.heappop(pq)

        if current in visited:
            continue

        visited.add(current)

        if current == end:
            return path, current_dist

        for neighbor, weight in graph.get(current, []):
            distance = current_dist + weight

            if neighbor not in visited and distance < distances.get(neighbor, float('inf')):
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor, path + [neighbor]))

    return None, float('inf')

# Example: Course prerequisites
graph = {
    'Python Basics': [('Data Structures', 2)],
    'Data Structures': [('Algorithms', 3), ('Machine Learning', 5)],
    'Algorithms': [('Machine Learning', 2)],
    'Machine Learning': []
}

path, distance = dijkstra(graph, 'Python Basics', 'Machine Learning')
# Returns: ['Python Basics', 'Data Structures', 'Algorithms', 'Machine Learning'], 7
```

**Complexity**: O((V + E) log V) with binary heap

**Mathematical Correctness**:
```
Proof by induction:
Base case: distance[start] = 0 (correct by definition)

Inductive step: When we visit node u, distance[u] is minimal because:
- All unvisited nodes have distance ≥ distance[u]
- All edges have positive weight
- Therefore, any path through unvisited nodes to u would be longer
```

**KG Applications**:
- Finding relationship chains with weighted importance
- Computing "semantic distance" between concepts
- Resource allocation in organizational graphs

#### 6. A* Search (Informed Shortest Path)

**Enhancement over Dijkstra**: Uses heuristic to guide search

**Formula**:
```
f(n) = g(n) + h(n)

where:
- g(n) = actual cost from start to n
- h(n) = heuristic estimate from n to goal
- f(n) = estimated total cost
```

**Admissible Heuristic**: h(n) ≤ actual cost (guarantees optimal solution)

**Example in KG**:
```python
def kg_heuristic(node, goal, embeddings):
    """
    Use embedding similarity as heuristic
    """
    return 1 - cosine_similarity(embeddings[node], embeddings[goal])

def astar_search(graph, start, goal, embeddings):
    pq = [(0 + kg_heuristic(start, goal, embeddings), 0, start, [start])]
    visited = set()

    while pq:
        f, g, current, path = heapq.heappop(pq)

        if current == goal:
            return path

        if current in visited:
            continue

        visited.add(current)

        for neighbor, cost in graph.get(current, []):
            if neighbor not in visited:
                new_g = g + cost
                new_h = kg_heuristic(neighbor, goal, embeddings)
                new_f = new_g + new_h
                heapq.heappush(pq, (new_f, new_g, neighbor, path + [neighbor]))

    return None
```

**Why This Works**:
- Combines graph structure (g) with semantic similarity (h)
- Explores promising paths first
- Faster than Dijkstra when heuristic is good

#### 7. Betweenness Centrality (Finding Bridges)

**Definition**: How often a node appears on shortest paths between other nodes

**Formula**:
```
C_B(v) = Σ σ_st(v) / σ_st
        s≠v≠t

where:
- σ_st = total number of shortest paths from s to t
- σ_st(v) = number of those paths passing through v
```

**Interpretation**:
- High betweenness = "bridge" node connecting communities
- Remove it → graph becomes disconnected or paths lengthen

**Algorithm** (Brandes' Algorithm):
```python
def betweenness_centrality(graph):
    """
    Compute betweenness centrality for all nodes
    """
    centrality = {node: 0.0 for node in graph}

    for source in graph:
        # BFS to find shortest paths
        stack = []
        paths = {node: [] for node in graph}  # predecessors
        sigma = {node: 0 for node in graph}  # number of shortest paths
        sigma[source] = 1
        dist = {node: -1 for node in graph}
        dist[source] = 0
        queue = [source]

        while queue:
            v = queue.pop(0)
            stack.append(v)
            for w in graph[v]:
                # First time visiting w?
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1

                # Shortest path to w via v?
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    paths[w].append(v)

        # Accumulate centrality
        delta = {node: 0 for node in graph}
        while stack:
            w = stack.pop()
            for v in paths[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != source:
                centrality[w] += delta[w]

    return centrality
```

**Complexity**: O(VE) for unweighted graphs

**KG Applications**:
- Identifying key connectors in organizational networks
- Finding critical concepts that link different domains
- Detecting information bottlenecks

**Example**:
```
Network:  A—B—C—D—E
          |     |
          F—G—H—I

Betweenness:
- B: High (bridges A and C-D-E)
- H: High (bridges F-G and I)
- C, D: Medium
- A, E, F, I: Low (endpoints)
```

#### 8. Graph Traversal with Constraints

**Pattern Matching in Cypher** (Extended):

```cypher
// Find paths with specific patterns
MATCH path = (a:Person)-[:WORKS_ON]->(p:Project)-[:REQUIRES]->(s:Skill)
WHERE a.name = 'Alice' AND s.category = 'Machine Learning'
RETURN path

// Variable-length paths (1 to 3 hops)
MATCH path = (a:Person)-[*1..3]-(b:Person)
WHERE a.name = 'Alice'
RETURN path, length(path)

// Shortest path with constraint
MATCH path = shortestPath((a:Person)-[*]-(b:Skill))
WHERE a.name = 'Alice' AND b.name = 'Python'
RETURN path

// All paths (warning: expensive!)
MATCH path = (a:Person)-[*]-(b:Person)
WHERE a.name = 'Alice' AND b.name = 'Bob'
AND length(path) <= 5
RETURN path
```

**Path Constraints**:
```cypher
// No repeated nodes (simple path)
MATCH path = (a)-[*]-(b)
WHERE a.name = 'Alice' AND b.name = 'Bob'
AND all(n IN nodes(path) WHERE size([x IN nodes(path) WHERE x = n]) = 1)
RETURN path

// No repeated relationships
MATCH path = (a)-[*]-(b)
WHERE a.name = 'Alice' AND b.name = 'Bob'
AND all(r IN relationships(path) WHERE size([x IN relationships(path) WHERE x = r]) = 1)
RETURN path

// Paths with specific relationship types
MATCH path = (a)-[:WORKS_FOR|MANAGES*]-(b)
WHERE a.name = 'Alice'
RETURN path
```

#### 9. Temporal Graph Algorithms

**Problem**: Graphs that change over time

**Example**: Social network where friendships form and break

**Temporal Reachability**:
```python
def temporal_path_exists(graph, start, end, start_time, end_time):
    """
    Check if path exists within time window where each edge
    timestamp is after the previous edge
    """
    # graph: {(u,v): timestamp}
    queue = [(start, start_time)]
    visited = set()

    while queue:
        node, time = queue.pop(0)

        if node == end and time <= end_time:
            return True

        if (node, time) in visited:
            continue

        visited.add((node, time))

        for neighbor in graph.get(node, []):
            edge_time = graph[(node, neighbor)]
            if edge_time >= time and edge_time <= end_time:
                queue.append((neighbor, edge_time))

    return False
```

**Applications**:
- "When did Alice first connect to Bob through collaborations?"
- "What skills did Alice acquire over time?"
- Version control of knowledge graphs

### Advanced Graph Properties

#### Graph Density

**Definition**: Ratio of actual edges to possible edges

```
Density = 2|E| / (|V|(|V|-1))  for undirected graphs
Density = |E| / (|V|(|V|-1))   for directed graphs
```

**Interpretation**:
- Density = 1: Complete graph (every node connected to every other)
- Density = 0: No edges
- Typical real graphs: 0.01 - 0.1 (sparse)

**KG Implications**:
- Very sparse KG → might have missing links
- Dense clusters → strong topical coherence
- Density varies by subgraph (heterogeneous structure)

#### Clustering Coefficient

**Definition**: How much nodes cluster together

**Local Clustering Coefficient** (for node v):
```
C(v) = (number of triangles connected to v) / (number of triples centered at v)
     = 2T(v) / (deg(v) · (deg(v)-1))

where T(v) = number of triangles including v
```

**Global Clustering Coefficient**:
```
C = 3 × (number of triangles) / (number of connected triples)
```

**Example**:
```
Graph:  A—B—C
        |/  |
        D—E—F

Triangles: {A,B,D}, {C,E,F}
C(B) = 2×1 / (3×2) = 1/3  (1 triangle, 3 neighbors)
C(E) = 2×1 / (3×2) = 1/3  (1 triangle, 3 neighbors)
C(A) = 2×1 / (2×1) = 1     (all neighbors connected)
```

**KG Applications**:
- High clustering → concepts form tightly-knit communities
- Low clustering → concepts are broadly distributed
- Measure of knowledge coherence

#### Graph Diameter and Average Path Length

**Diameter**: Maximum shortest path between any two nodes

```
diameter(G) = max dist(u,v)
              u,v∈V
```

**Average Path Length**:
```
L = (1 / (|V|(|V|-1))) Σ dist(u,v)
                       u≠v
```

**Small-World Property**:
- High clustering coefficient
- Low average path length
- L ∝ log(|V|)

**Example**: Social networks ("six degrees of separation")

**KG Implications**:
- Small diameter → knowledge is well-connected
- Large diameter → fragmented knowledge
- Informs retrieval strategies (how many hops to explore)

### Graph Embeddings (Bridging Graphs and Vectors)

**Goal**: Represent nodes as vectors while preserving graph structure

**Node2Vec**: Graph version of Word2Vec

**Idea**:
1. Generate random walks from each node
2. Treat walks as "sentences"
3. Apply Word2Vec

**DeepWalk**:
```
Random walk from A: A → C → D → E → C
Treat as: "A C D E C"
Learn embeddings so nearby nodes in walks are close in vector space
```

**Graph Convolutional Networks (GCN)**:

**Message Passing**:
```
h_v^(k+1) = σ(W^(k) · Σ h_u^(k) / |N(v)|)
                      u∈N(v)

Translation: "Update node v's embedding by aggregating its neighbors"
```

**Why this matters for RAG+KG**:
- Can use graph embeddings for similarity search
- Combine with text embeddings for hybrid retrieval
- Find similar entities even without direct text match

---

## Semantic Similarity Theory (The Mathematics of Meaning Comparison)

### Distance vs. Similarity

**Distance**: How far apart are two points?
- Euclidean: Straight-line distance
- Manhattan: Sum of coordinate differences
- **Smaller = more similar**

**Similarity**: How alike are two points?
- Cosine: Angle between vectors
- Dot product: Alignment
- **Larger = more similar**

**Relationship**:
```
similarity = 1 / (1 + distance)
or
distance = arccos(similarity)
```

### Metrics Deep Dive

#### 1. Cosine Similarity (Revisited)

```
cos(θ) = (u · v) / (||u|| ||v||)

Range: [-1, 1]
```

**When to use**: Text embeddings (scale-invariant)

**Properties**:
- Triangle inequality holds
- Invariant to vector magnitude
- Perfect for embeddings (normalized vectors)

#### 2. Euclidean Distance

```
d(u,v) = √(Σ(uᵢ - vᵢ)²)

Range: [0, ∞)
```

**When to use**: When magnitude matters (image embeddings, spatial data)

**Intuition**:
```
2D space:
u = (1,2), v = (4,6)
d = √((4-1)² + (6-2)²) = √(9+16) = 5
```

#### 3. Dot Product

```
u · v = Σ uᵢvᵢ = ||u|| ||v|| cos(θ)

Range: (-∞, ∞)
```

**When to use**: When both magnitude and angle matter

**In neural networks**: Used before softmax
```
attention_score = Q · K^T
```

#### 4. Jaccard Similarity (For Sets)

```
J(A,B) = |A ∩ B| / |A ∪ B|

Range: [0, 1]
```

**Example**:
```
Doc A words: {cat, dog, mouse}
Doc B words: {dog, mouse, rat}

J(A,B) = |{dog, mouse}| / |{cat, dog, mouse, rat}| = 2/4 = 0.5
```

**Use case**: Document deduplication, fuzzy matching

#### 5. Edit Distance (Levenshtein)

**Number of edits to transform string A → B**

**Operations**: Insert, delete, substitute

**Example**:
```
"kitten" → "sitting"
1. kitten → sitten  (substitute k→s)
2. sitten → sittin  (substitute e→i)
3. sittin → sitting (insert g)

Edit distance = 3
```

**Algorithm** (Dynamic Programming):
```
D[i,j] = min(
    D[i-1,j] + 1,      # deletion
    D[i,j-1] + 1,      # insertion
    D[i-1,j-1] + cost  # substitution (cost=0 if match)
)
```

**Use case**: Spell checking, entity matching

### Similarity in Context: The Attention Mechanism

**The Problem**: Not all words are equally important for similarity.

**Query**: "capital of France"
**Doc**: "Paris is the beautiful capital of France, known for the Eiffel Tower"

**Which words matter most?** capital, France, Paris (not "beautiful", "known")

**Attention Solution**:

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

where:
- Q = query vector
- K = key vectors (from words)
- V = value vectors (from words)
- d_k = dimension (scaling factor)
```

**Step by step**:

1. **Compute scores**: How much does query attend to each word?
   ```
   scores = Q · [K₁, K₂, K₃, ...]
   scores = [score₁, score₂, score₃, ...]
   ```

2. **Normalize** (softmax): Convert to probabilities
   ```
   attention_weights = softmax(scores / √d_k)
   ```

3. **Weighted sum**: Combine values using attention weights
   ```
   output = Σ attention_weights_i · V_i
   ```

**Visual**:
```
Query: "capital France"

Attention weights on doc:
"Paris":    0.40  ████████
"is":       0.02  ▌
"the":      0.01  ▌
"capital":  0.35  ███████
"of":       0.02  ▌
"France":   0.20  ████
```

**Result**: Output emphasizes Paris, capital, France

**Why √d_k scaling?**

Without scaling, dot products grow with dimension:
```
d=100:  Q·K ~ 100 * σ²  (large variance)
d=1000: Q·K ~ 1000 * σ² (very large!)

Result: softmax saturates (all weight on one token)

With scaling: Q·K/√d ~ √d * σ² (controlled variance)
```

### Semantic Similarity for RAG

**Query-Document Matching**:

**Level 1: Word Overlap** (BM25)
- Counts matching words
- Ignores synonyms

**Level 2: Embedding Similarity** (Dense Retrieval)
- Captures semantic meaning
- "automobile" matches "car"

**Level 3: Cross-Attention** (Reranker)
- Word-by-word comparison
- "capital of France" explicitly attends to "Paris"

**Progressive Refinement**:
```
Stage 1 (BM25):       1000 docs → 100 candidates   (fast, recall-focused)
Stage 2 (Dense):      100 docs → 20 candidates     (medium, balance)
Stage 3 (Cross-Enc):  20 docs → 5 final docs       (slow, precision-focused)
```

**Why this cascade works**:
- Each stage gets more expensive but more accurate
- Reduces expensive computation to small candidate set
- Combines complementary strengths

---

This theoretical foundation transforms your understanding from "it works" to "I know why it works and when it will fail." Continue to Section 3 for applying these principles in production RAG systems!

(If you made it through Section 2A, congratulations - you now know more about the math behind RAG than most people deploying it to production. If you skipped 2A, that's fine too. But when your retrieval quality mysteriously degrades or your vector search returns nonsense, come back and read the theory. It will make sense the second time, when you have real failures to map to the concepts.)

---
