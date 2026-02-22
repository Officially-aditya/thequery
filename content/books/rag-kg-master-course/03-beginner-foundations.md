# 2. BEGINNER-FRIENDLY FOUNDATIONS

(This section builds your mental model of how everything works under the hood. If you already know LLMs, embeddings, and graphs, you might be tempted to skip this. Don't. The nuances here matter for production systems, and we'll highlight failure modes that aren't obvious until you've shipped broken code to users.)

## LLM Basics (Large Language Models)

### What is an LLM?

**Simple Analogy**: Imagine a super-smart autocomplete that has read most of the internet. You give it a prompt, it predicts the most likely continuation.

**Technical Definition**: A neural network trained on vast text data to predict the next token (word/subword) given previous context.

### Core Concepts You Must Understand

#### 2.1.1 Tokens & Tokenization

**What**: Text is broken into chunks (tokens) before processing.

```python
# Example: How text becomes tokens
text = "RAG systems are powerful"

# GPT tokenization (simplified)
tokens = ["RAG", " systems", " are", " powerful"]
token_ids = [22060, 6067, 527, 8147]  # Numeric IDs

# Why it matters:
# - APIs charge per token
# - Models have token limits (8k, 32k, 128k)
# - 1 token ≈ 0.75 words on average
```

**Key Insight**: "Hello world" = 2 tokens, but "Supercalifragilisticexpialidocious" might be 5+ tokens.

#### 2.1.2 Embeddings

**What**: Converting text into dense vector representations (arrays of numbers) that capture semantic meaning.

**Analogy**: Like coordinates on a map, but instead of (latitude, longitude), you have 1536 dimensions representing meaning.

```python
from openai import OpenAI
client = OpenAI()

# Create an embedding
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Knowledge graphs organize information"
)

embedding = response.data[0].embedding
# Result: [0.023, -0.15, 0.087, ..., 0.032]  # 1536 numbers
# Length: 1536 dimensions

# Similar sentences have similar embeddings
embedding_2 = client.embeddings.create(
    model="text-embedding-3-small",
    input="Graphs structure knowledge"
).data[0].embedding

# Cosine similarity will be high (0.85+)
```

**Why Embeddings Matter for RAG**:
- They enable semantic search ("find similar meaning" not just keyword matching)
- Power vector databases
- Core of retrieval systems

**Visualization**:
```
Text Space:              Embedding Space (simplified to 2D):
"Dog"                    (0.8, 0.6)  ●
"Cat"                    (0.75, 0.65) ●  <- Close to dog
"Knowledge Graph"        (-0.2, 0.9)        ●
"Graph Database"         (-0.15, 0.85)      ● <- Close to KG

Distance = Semantic Similarity
```

#### 2.1.3 Prompting Fundamentals

**Zero-Shot Prompting**:
```
Prompt: "Translate to French: Hello"
Response: "Bonjour"
```

**Few-Shot Prompting**:
```
Prompt:
"
Translate to French:
Hello -> Bonjour
Goodbye -> Au revoir
Thank you -> ?
"
Response: "Merci"
```

**Structured Prompting** (Critical for RAG):
```python
prompt = f"""
You are a helpful assistant that answers questions using provided context.

Context:
{retrieved_documents}

Question: {user_question}

Instructions:
- Only use information from the context
- If the answer isn't in the context, say "I don't know"
- Cite the source document

Answer:
"""
```

### 2.1.4 Transformer Architecture Deep Dive

**Why Transformers Matter for RAG**: Understanding transformer architecture helps you choose the right models, optimize inference, and debug issues in production RAG systems.

(This subsection gets mathematical. That's unavoidable - transformers are the engine under the hood of everything you'll build. You don't need to memorize the equations, but you should understand what each component does and why. If your eyes glaze over during the attention mechanism explanation, that's normal. Come back to it later when you're debugging why your retrieval is slow.)

#### The Transformer Revolution

**Before Transformers (Pre-2017)**:
- **RNNs/LSTMs**: Sequential processing, slow, can't parallelize
- **Limited context**: Struggled with long-range dependencies
- **No bidirectional context**: Hard to capture full semantic meaning

**After Transformers (2017-present)**:
- **Parallel processing**: All tokens processed simultaneously
- **Self-attention**: Every token can attend to every other token
- **Scalability**: Can be trained on massive datasets efficiently

#### Core Components of a Transformer

**1. Input Embedding Layer**

Text → Tokens → Embeddings:
```
"The cat sat" → [501, 2368, 3287] (token IDs)
              → [[0.2, -0.5, ...], [0.1, 0.8, ...], [-0.3, 0.2, ...]]
              (d-dimensional vectors, typically d=768 or 1536)
```

**2. Positional Encoding**

**Problem**: Self-attention is position-invariant ("cat sat" = "sat cat")

**Solution**: Add positional information to embeddings

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

where:
- pos = position in sequence
- i = dimension index
- d = embedding dimension
```

**Why sine/cosine?**:
- **Bounded**: Values stay in [-1, 1]
- **Unique**: Each position gets unique encoding
- **Relative positioning**: PE(pos+k) can be expressed as linear function of PE(pos)
- **Extrapolation**: Can handle sequences longer than seen in training

**Alternative**: Learned positional embeddings (used in BERT, GPT)

**3. Multi-Head Self-Attention**

**Single Attention Head**:
```
Input: X ∈ ℝ^(n×d)  (n tokens, d dimensions each)

1. Project to Q, K, V:
   Q = XW_Q,  K = XW_K,  V = XW_V
   where W_Q, W_K, W_V ∈ ℝ^(d×d_k)

2. Compute attention scores:
   Attention(Q,K,V) = softmax(QK^T / √d_k) V

Step-by-step:
   - QK^T: n×n matrix of all-pairs dot products (how relevant is each token to each other?)
   - / √d_k: Scale to prevent vanishing gradients
   - softmax: Normalize to probabilities (each row sums to 1)
   - × V: Weighted sum of value vectors
```

**Why √d_k Scaling?**

Without scaling, for large d_k:
```
If Q,K have zero mean and unit variance:
QK^T has variance d_k

For d_k = 512: dot products are in range [-30, 30]
softmax([-30, -5, 0, 5, 30]) ≈ [0, 0, 0, 0, 1]  ← All weight on one token!

With scaling by √d_k:
QK^T / √d_k has variance 1
softmax([-4.2, -0.7, 0, 0.7, 4.2]) ≈ [0.01, 0.12, 0.24, 0.48, 0.15]  ← Better distribution!
```

(This scaling factor might seem like a minor detail. It's not. Without it, attention collapses to mostly zeros and ones, and your model learns nothing. This is one of those "the devil is in the details" moments that separates working code from broken code.)

**Multi-Head Attention**:

Instead of one attention, use h parallel heads:

```
MultiHead(Q,K,V) = Concat(head₁, head₂, ..., head_h) W_O

where head_i = Attention(QW_Qⁱ, KW_Kⁱ, VW_Vⁱ)
```

**Why Multiple Heads?**:
- **Different relationships**: Head 1 might capture syntax, Head 2 semantics, Head 3 coreference
- **Different subspaces**: Each head operates in different d_k-dimensional subspace
- **Ensemble effect**: Combining heads gives robust representation

**Example**:
```
Sentence: "The cat sat on the mat"

Head 1 (Syntax):        Head 2 (Semantics):     Head 3 (Reference):
"cat" → "sat" (0.8)     "cat" → "mat" (0.6)     "cat" → "The" (0.7)
(subject-verb)          (agent-location)         (noun-determiner)
```

**4. Feed-Forward Networks**

After attention, each position passes through identical FFN:

```
FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂

where:
- W₁ ∈ ℝ^(d_model × d_ff), typically d_ff = 4 × d_model
- W₂ ∈ ℝ^(d_ff × d_model)
```

**Why FFN?**:
- Attention captures relationships, FFN adds non-linearity and expressiveness
- Each position processed independently (no mixing across positions)
- Huge parameter count (most parameters are here!)

**5. Layer Normalization & Residual Connections**

```
# After attention
x = LayerNorm(x + MultiHeadAttention(x))

# After FFN
x = LayerNorm(x + FFN(x))
```

**LayerNorm**:
```
LayerNorm(x) = γ ⊙ (x - μ) / σ + β

where:
- μ = mean(x)
- σ = std(x)
- γ, β = learned parameters
```

**Why This Matters**:
- **Residual connections**: Prevent vanishing gradients in deep networks (GPT-3 has 96 layers!)
- **Layer norm**: Stabilizes training, allows higher learning rates

#### Encoder vs. Decoder Transformers

**Encoder** (BERT):
```
Input: Full sentence
Attention: Bidirectional (each token sees all tokens)
Output: Contextualized representation of each token
Use case: Understanding, classification, embedding
```

**Decoder** (GPT):
```
Input: Prefix of sequence
Attention: Causal/Masked (token i can only see tokens ≤ i)
Output: Probability distribution for next token
Use case: Generation, completion
```

**Encoder-Decoder** (T5, BART):
```
Encoder: Process input
Decoder: Generate output, attending to encoder
Use case: Translation, summarization
```

**Causal Masking in Decoders**:
```
Attention matrix without mask:
     t1   t2   t3   t4
t1 [0.2  0.3  0.1  0.4]
t2 [0.1  0.4  0.2  0.3]
t3 [0.3  0.1  0.5  0.1]
t4 [0.2  0.2  0.2  0.4]

With causal mask (zero out future):
     t1   t2   t3   t4
t1 [1.0  0    0    0  ]
t2 [0.3  0.7  0    0  ]
t3 [0.2  0.1  0.7  0  ]
t4 [0.2  0.2  0.2  0.4]

This prevents t2 from "cheating" by looking at t3, t4
```

#### Complete Transformer Block

```
Input: Token embeddings + Positional encodings
  ↓
Multi-Head Attention
  ↓
Add & Norm (residual + layer norm)
  ↓
Feed-Forward Network
  ↓
Add & Norm
  ↓
Output: Contextualized representations

×  N layers (N=12 for BERT-base, N=96 for GPT-3)
```

#### Key Parameters and Model Sizes

**BERT-base**:
- Layers: 12
- Hidden size: 768
- Attention heads: 12
- Parameters: 110M
- Context window: 512 tokens

**BERT-large**:
- Layers: 24
- Hidden size: 1024
- Attention heads: 16
- Parameters: 340M

**GPT-3**:
- Layers: 96
- Hidden size: 12,288
- Attention heads: 96
- Parameters: 175B
- Context window: 2048 tokens

**GPT-4** (estimated):
- Parameters: ~1.76T (mixture of experts)
- Context window: 128K tokens

#### Computational Complexity

**Self-Attention**: O(n² · d)
- n = sequence length
- d = embedding dimension
- Bottleneck for long sequences!

**Why This Matters for RAG**:
- Long documents → expensive to embed
- Chunking reduces n → manageable computation
- Context window limits affect retrieval design

**Approximations for Long Sequences**:
1. **Sparse attention** (BigBird, Longformer): O(n · log n)
2. **Linear attention**: O(n · d²)
3. **Chunking**: Process in windows (used in RAG!)

#### Inference Optimization for RAG

(Most tutorials skip these optimizations. Then you deploy to production and wonder why your RAG system costs $10,000/month and takes 5 seconds per query. Read this section carefully. Your AWS bill will thank you.)

**KV Caching**:

When generating tokens autoregressively:
```
# Without KV cache:
t1: compute attention for "The"
t2: recompute attention for "The", compute for "cat"
t3: recompute for "The", "cat", compute for "sat"
→ O(n²) redundant computation!

# With KV cache:
t1: compute K,V for "The", cache them
t2: reuse K,V for "The", compute only for "cat"
t3: reuse K,V for "The", "cat", compute only for "sat"
→ O(n) computation, huge speedup!
```

**Batch Processing**:
```python
# Inefficient: one at a time
for doc in documents:
    embedding = model.encode(doc)  # Separate forward pass

# Efficient: batched
embeddings = model.encode(documents, batch_size=32)  # One forward pass
# 10-100x faster!
```

## Retrieval Basics

### What is Retrieval?

**Goal**: Given a query, find the most relevant documents from a large collection.

**Real-World Analogy**:
- Google Search = Retrieval system
- Library catalog = Retrieval system
- Your brain searching memories = Retrieval system

### Types of Retrieval

#### 2.2.1 Keyword Search (BM25)

**How it works**: Count matching words, adjust for document length and term rarity.

```python
from rank_bm25 import BM25Okapi

documents = [
    "Knowledge graphs represent structured information",
    "RAG combines retrieval with generation",
    "Vector databases enable semantic search"
]

# Tokenize
tokenized_docs = [doc.split() for doc in documents]

# Build BM25 index
bm25 = BM25Okapi(tokenized_docs)

# Query
query = "semantic search"
scores = bm25.get_scores(query.split())
# [0.2, 0.1, 0.9] <- Document 3 wins
```

**Strengths**: Fast, works with exact matches, no ML needed
**Weaknesses**: Misses semantic similarity ("car" vs "automobile")

#### 2.2.2 Semantic Search (Dense Retrieval)

(You might be thinking "BM25 is old-school, I'll just use embeddings for everything." Please don't. BM25 beats semantic search for exact phrase matching, rare technical terms, and names. This is why we combine them in hybrid retrieval - and why ignoring BM25 will haunt you when users search for product SKUs or error codes.)

**How it works**: Convert query and documents to embeddings, find nearest neighbors.

```python
import numpy as np
from openai import OpenAI

client = OpenAI()

def get_embedding(text):
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

# Embed documents
docs = [
    "Knowledge graphs represent structured information",
    "RAG combines retrieval with generation",
    "Vector databases enable semantic search"
]
doc_embeddings = [get_embedding(doc) for doc in docs]

# Embed query
query = "what is semantic search?"
query_embedding = get_embedding(query)

# Calculate cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

scores = [cosine_similarity(query_embedding, doc_emb)
          for doc_emb in doc_embeddings]
# [0.65, 0.58, 0.91] <- Document 3 wins (semantic match!)
```

**Strengths**: Understands meaning, handles synonyms
**Weaknesses**: Slower, requires embeddings, can miss exact matches

#### 2.2.3 Hybrid Retrieval (Best of Both)

```python
# Combine BM25 + Semantic
bm25_scores = normalize(bm25.get_scores(query))
semantic_scores = normalize(cosine_similarities)

# Weighted combination
final_scores = 0.3 * bm25_scores + 0.7 * semantic_scores
```

### Vector Databases

**What**: Specialized databases optimized for storing and searching embeddings.

**Key Operations**:
1. **Insert**: Store vectors with metadata
2. **Search**: Find k-nearest neighbors (kNN)
3. **Filter**: Combine vector search with metadata filters

```python
import chromadb

# Initialize
client = chromadb.Client()
collection = client.create_collection("my_docs")

# Add documents
collection.add(
    documents=["RAG is powerful", "KG structures knowledge"],
    ids=["doc1", "doc2"],
    metadatas=[{"source": "paper1"}, {"source": "paper2"}]
)

# Query
results = collection.query(
    query_texts=["what is RAG?"],
    n_results=2
)
# Returns: most similar documents
```

**Popular Vector DBs**:
- **FAISS**: Fast, local, Facebook's library
- **ChromaDB**: Simple, embedded, great for prototyping
- **Pinecone**: Managed, production-grade, scales automatically
- **Weaviate**: Open-source, full-featured

## Knowledge Graph Fundamentals

### What is a Knowledge Graph?

**Definition**: A graph-structured database where knowledge is stored as entities (nodes) and relationships (edges).

**Real-World Analogy**:
- Social network (Facebook): People = nodes, Friendships = edges
- Map: Cities = nodes, Roads = edges
- Knowledge: Concepts = nodes, Relationships = edges

### Core Components

#### Nodes (Entities)
Things that exist: Person, Company, Product, Concept

#### Edges (Relationships)
How things connect: WORKS_FOR, OWNS, IS_PART_OF

#### Properties
Attributes of nodes/edges: name, age, date, weight

### Graph Representation

**Visual**:
```
(Person:Alice {age: 30})
       |
       | -[WORKS_FOR {since: 2020}]->
       |
       v
(Company:Acme {industry: "Tech"})
```

**Triple Format** (Subject-Predicate-Object):
```
Alice WORKS_FOR Acme
Alice AGE 30
Acme INDUSTRY "Tech"
```

**Why Graphs Beat Tables**:

(This is the most common question: "Why not just use PostgreSQL?" Fair question. Short answer: for simple lookups, you should. But try expressing "find friends-of-friends who work at competitors of companies in my portfolio" in SQL. You'll end up with 5 self-joins and a query planner that gives up. Graphs shine for traversals and multi-hop queries. Everything else, use the tool you already know.)

**Relational Database (Tables)**:
```
Employees Table:
| ID | Name  | Company | Age |
|----|-------|---------|-----|
| 1  | Alice | Acme    | 30  |

Companies Table:
| Name | Industry |
|------|----------|
| Acme | Tech     |

# To find "Who works in Tech?":
# Need JOIN operation - slow for complex queries
```

**Knowledge Graph**:
```
MATCH (p:Person)-[:WORKS_FOR]->(c:Company {industry: "Tech"})
RETURN p.name

# Direct traversal - fast even with millions of nodes
```

### Graph Theory Intuition

#### Paths
Sequence of connected nodes:
```
Alice -> WORKS_FOR -> Acme -> LOCATED_IN -> San Francisco
```

#### Multi-Hop Queries
Follow multiple relationships:
```
"Find friends of friends who work at tech companies"
(Me)-[:FRIEND]->(Friend)-[:FRIEND]->(FoF)-[:WORKS_FOR]->(Company {industry: "Tech"})
```

#### Neighborhoods
All nodes within N steps:
```
# 1-hop neighborhood of Alice
Alice -> Acme, Bob, Project_X

# 2-hop neighborhood
Alice -> Acme -> [All employees], Bob -> [Bob's friends], ...
```

## Cypher & SPARQL Basics

### Cypher (Neo4j Query Language)

**ASCII Art Syntax** - Intuitive and visual!

```cypher
// Create nodes
CREATE (a:Person {name: "Alice", age: 30})
CREATE (c:Company {name: "Acme"})

// Create relationship
CREATE (a)-[:WORKS_FOR {since: 2020}]->(c)

// Query: Find all people working at Acme
MATCH (p:Person)-[:WORKS_FOR]->(c:Company {name: "Acme"})
RETURN p.name, p.age

// Multi-hop: Friends of Alice who work in Tech
MATCH (alice:Person {name: "Alice"})-[:FRIEND]-(friend)-[:WORKS_FOR]->(c:Company {industry: "Tech"})
RETURN friend.name, c.name

// Aggregation: Count employees per company
MATCH (p:Person)-[:WORKS_FOR]->(c:Company)
RETURN c.name, COUNT(p) AS employee_count
ORDER BY employee_count DESC
```

### SPARQL (RDF Query Language)

**Used for**: Semantic web, ontologies, Wikidata

```sparql
# Find all companies Alice works for
SELECT ?company WHERE {
    :Alice :worksFor ?company .
}

# Multi-hop
SELECT ?friendCompany WHERE {
    :Alice :friend ?friend .
    ?friend :worksFor ?friendCompany .
}
```

**For this course**: We'll focus on **Cypher** (more popular in industry).

---
