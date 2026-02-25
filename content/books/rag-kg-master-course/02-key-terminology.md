# 1.5 KEY TERMINOLOGY & DEFINITIONS

> **Purpose**: This comprehensive glossary defines all technical terms, acronyms, and concepts used throughout the course. Reference this section whenever you encounter unfamiliar terminology.

(Don't try to memorize all of this now. That would be a waste of time. Skim it once to see what's here, then come back when you need it. Think of this as a dictionary, not a textbook chapter. Nobody reads dictionaries cover-to-cover for a reason.)

## Core Acronyms & Abbreviations

### A-E

**ANN** (Approximate Nearest Neighbor): Algorithm for finding points in a dataset that are closest to a query point, with some tolerance for error in exchange for speed. Used in vector databases for efficient similarity search.

**API** (Application Programming Interface): A set of protocols and tools that allow different software applications to communicate with each other.

**BERT** (Bidirectional Encoder Representations from Transformers): A transformer-based language model developed by Google that processes text bidirectionally (looking at both left and right context simultaneously).

**BFS** (Breadth-First Search): A graph traversal algorithm that explores all neighbors at the current depth before moving to nodes at the next depth level.

**BM25** (Best Matching 25): A ranking function used in information retrieval to estimate the relevance of documents to a given search query, based on term frequency and document length.

**CBOW** (Continuous Bag of Words): A word embedding model that predicts a target word from its surrounding context words.

**Cypher**: A declarative graph query language created for Neo4j, using ASCII-art syntax to represent graph patterns.

**DAG** (Directed Acyclic Graph): A directed graph with no cycles - you cannot start at a node and follow directed edges back to that same node.

**DFS** (Depth-First Search): A graph traversal algorithm that explores as far as possible along each branch before backtracking.

**DPR** (Dense Passage Retrieval): A neural retrieval method that encodes queries and documents as dense vectors for similarity-based retrieval.

**Embedding**: A learned, dense vector representation of data (text, images, graphs) in a continuous vector space where semantically similar items are close together.

**EHR** (Electronic Health Records): Digital version of a patient's paper chart, containing medical history, diagnoses, medications, and treatment plans.

### F-M

**FAISS** (Facebook AI Similarity Search): A library developed by Meta for efficient similarity search and clustering of dense vectors, optimized for billion-scale datasets.

**FFN** (Feed-Forward Network): A neural network layer where information moves in only one direction, from input through hidden layers to output, with no cycles.

**GCN** (Graph Convolutional Network): A type of neural network that operates on graph-structured data by aggregating information from node neighborhoods.

**GPT** (Generative Pre-trained Transformer): A series of large language models developed by OpenAI that use decoder-only transformer architecture for text generation.

**Hallucination**: When an LLM generates information that sounds plausible but is factually incorrect or not grounded in the provided context.

**IDF** (Inverse Document Frequency): A measure of how much information a word provides - rare words have high IDF, common words have low IDF.

**kNN** (k-Nearest Neighbors): An algorithm that finds the k closest points to a query point in a dataset, used for classification, regression, or retrieval.

**KG** (Knowledge Graph): A structured representation of knowledge as entities (nodes) and relationships (edges), often with properties attached to both.

**LLM** (Large Language Model): A neural network with billions of parameters trained on vast amounts of text data to understand and generate human-like text.

**LSA** (Latent Semantic Analysis): A technique for analyzing relationships between documents and terms using singular value decomposition of term-document matrices.

### N-Z

**NER** (Named Entity Recognition): The task of identifying and classifying named entities (people, organizations, locations, etc.) in text.

**NLP** (Natural Language Processing): A field of AI focused on enabling computers to understand, interpret, and generate human language.

**Ontology**: A formal specification of concepts and relationships within a domain, defining what things exist and how they relate.

**PageRank**: An algorithm that measures the importance of nodes in a graph based on the structure of incoming links, originally developed for ranking web pages.

**RAG** (Retrieval-Augmented Generation): A technique that combines information retrieval with text generation - retrieve relevant documents, then generate answers based on them.

**RDF** (Resource Description Framework): A framework for representing information about resources in the web, using subject-predicate-object triples.

**Reranking**: A second-stage ranking process that reorders initially retrieved results using more sophisticated (and computationally expensive) relevance signals.

**RNN** (Recurrent Neural Network): A neural network architecture designed for sequential data, where outputs from previous steps feed back as inputs.

**SPARQL**: A query language for RDF databases, similar to SQL but designed for graph-structured data.

**TF** (Term Frequency): A measure of how frequently a term appears in a document.

**TF-IDF** (Term Frequency-Inverse Document Frequency): A numerical statistic that reflects how important a word is to a document in a collection, balancing term frequency against rarity.

**Transformer**: A neural network architecture based on self-attention mechanisms that processes all positions of a sequence simultaneously, enabling parallelization.

**Vector Database**: A specialized database optimized for storing and querying high-dimensional vector embeddings, supporting operations like similarity search.

---

## Fundamental Concepts Defined

### Embeddings & Vector Representations

**Embedding**: A learned mapping from discrete objects (words, sentences, documents, nodes) to continuous vector spaces.

- **Formal Definition**: A function f: X → ℝ^d that maps items from space X to d-dimensional real-valued vectors
- **Example**: The word "king" → [0.23, -0.41, 0.87, ..., 0.15] (768 dimensions)
- **Purpose**: Convert categorical/symbolic data into numerical form suitable for machine learning
- **Property**: Semantically similar items should have similar (high cosine similarity) embeddings

**Dense Vector**: A vector where most/all elements are non-zero, as opposed to sparse vectors.

**Sparse Vector**: A vector where most elements are zero (e.g., one-hot encoding, TF-IDF with large vocabulary).

**Dimensionality**: The number of elements in a vector. Common embedding dimensions: 128, 256, 384, 768, 1536, 3072.

**Vector Space**: A mathematical structure where vectors can be added together and multiplied by scalars, with defined operations like dot product and norm.

**Semantic Similarity**: The degree to which two pieces of text have similar meaning, often measured by cosine similarity of their embeddings.

**Cosine Similarity**: A measure of similarity between two vectors based on the cosine of the angle between them, ranging from -1 (opposite) to 1 (identical direction).
```
cos(θ) = (A·B) / (||A|| ||B||)
```

**Euclidean Distance**: The straight-line distance between two points in vector space.
```
d(A,B) = √(Σ(Aᵢ - Bᵢ)²)
```

**Manhattan Distance**: The sum of absolute differences between vector components, like distance traveled on a grid.
```
d(A,B) = Σ|Aᵢ - Bᵢ|
```

**Dot Product**: The sum of element-wise products of two vectors, related to both their magnitude and angle.
```
A·B = Σ AᵢBᵢ
```

**Norm**: The length/magnitude of a vector.
```
||A|| = √(Σ Aᵢ²)  (L2 norm)
||A|| = Σ |Aᵢ|     (L1 norm)
```

---

### Language Model Concepts

**Token**: The basic unit of text that a language model processes. Can be a word, subword, or character depending on the tokenization scheme.

**Tokenization**: The process of breaking text into tokens.
- **Word-level**: "Hello world" → ["Hello", "world"]
- **Subword-level**: "unhappiness" → ["un", "happiness"]
- **Character-level**: "Hi" → ["H", "i"]

**Vocabulary**: The set of all possible tokens a model can recognize. Typical sizes: 32K-100K tokens.

**Context Window**: The maximum number of tokens a model can process at once. Examples:
- GPT-3: 2,048 tokens
- GPT-4: 8,192 tokens (GPT-4-32K: 32,768 tokens)
- Claude 3: 200,000 tokens

**Prompt**: The input text provided to a language model to elicit a response.

**Completion**: The text generated by a language model in response to a prompt.

**Zero-Shot Learning**: A model performing a task without any examples, using only instructions.

**Few-Shot Learning**: A model performing a task given a few examples in the prompt.

**Fine-Tuning**: Further training a pre-trained model on specific data to adapt it to a particular task or domain.

**Temperature**: A parameter controlling randomness in generation. Lower (0.0-0.3) = more deterministic, higher (0.7-1.0) = more creative.

**Top-k Sampling**: Limiting token selection to the k most likely next tokens.

**Top-p Sampling** (Nucleus Sampling): Selecting from the smallest set of tokens whose cumulative probability exceeds p.

**Attention**: A mechanism that allows models to focus on different parts of the input when processing each element.

**Self-Attention**: Attention where a sequence attends to itself, allowing each position to gather information from all other positions.

**Multi-Head Attention**: Running multiple attention mechanisms in parallel, each learning different relationship patterns.

**Query, Key, Value** (Q, K, V): The three projections used in attention mechanisms:
- **Query**: "What am I looking for?"
- **Key**: "What information do I contain?"
- **Value**: "What information do I communicate?"

**Attention Score**: The computed relevance between a query and each key, determining how much each value contributes.

**Layer Normalization**: A technique that normalizes activations across features for each sample, stabilizing training.

**Residual Connection**: A shortcut connection that adds the input of a layer to its output, helping gradient flow in deep networks.

---

### Retrieval Concepts

**Information Retrieval** (IR): The process of finding relevant documents from a large collection based on a query.

**Query**: The user's information need expressed as text (in RAG systems).

**Document**: A unit of retrievable content (can be a full document, paragraph, or chunk).

**Chunking**: Dividing large documents into smaller, semantically coherent pieces for embedding and retrieval.

**Chunk**: A segment of a document, typically 100-1000 tokens, treated as a single retrievable unit.

**Overlap**: The number of tokens shared between consecutive chunks to preserve context at boundaries.

**Retrieval**: The process of finding and ranking relevant chunks/documents for a query.

**Ranking**: Ordering retrieved results by relevance to the query.

**Relevance**: How well a document satisfies the information need expressed in a query.

**Precision**: The fraction of retrieved documents that are relevant.
```
Precision = (Relevant Retrieved) / (Total Retrieved)
```

**Recall**: The fraction of relevant documents that were retrieved.
```
Recall = (Relevant Retrieved) / (Total Relevant)
```

**F1 Score**: The harmonic mean of precision and recall.
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Top-k Retrieval**: Returning only the k most relevant results.

**Recall@k**: The fraction of relevant documents found in the top k results.

**MRR** (Mean Reciprocal Rank): Average of reciprocal ranks of the first relevant result.
```
MRR = (1/|Q|) Σ 1/rankᵢ
```

**Dense Retrieval**: Using learned dense vector embeddings for retrieval (semantic search).

**Sparse Retrieval**: Using sparse representations like TF-IDF or BM25 (keyword search).

**Hybrid Retrieval**: Combining dense and sparse retrieval methods.

**Lexical Match**: Matching based on exact word overlap between query and document.

**Semantic Match**: Matching based on meaning, even if different words are used.

**Cross-Encoder**: A model that jointly encodes query and document to compute relevance (slow but accurate).

**Bi-Encoder**: A model that separately encodes query and document (fast, used for initial retrieval).

**Reranker**: A model (often cross-encoder) that reorders initially retrieved results for better precision.

**Hard Negatives**: Negative examples (non-relevant documents) that are similar to positive examples, used to train better retrievers.

---

### Graph Concepts

**Graph**: A mathematical structure G = (V, E) consisting of vertices (nodes) and edges (connections).

**Node** (Vertex): An entity in a graph (e.g., Person, Company, Concept).

**Edge** (Relationship, Link): A connection between two nodes.

**Directed Graph** (Digraph): A graph where edges have direction (A→B is different from B→A).

**Undirected Graph**: A graph where edges are bidirectional (A-B means both directions).

**Weighted Graph**: A graph where edges have associated weights/values.

**Property Graph**: A graph where nodes and edges can have multiple key-value properties.

**Label**: A type or category for nodes or edges (e.g., :Person, :WORKS_FOR).

**Degree**: The number of edges connected to a node.
- **In-degree**: Number of incoming edges
- **Out-degree**: Number of outgoing edges

**Path**: A sequence of nodes connected by edges.

**Path Length**: The number of edges in a path.

**Shortest Path**: The path with minimum length between two nodes.

**Cycle**: A path that starts and ends at the same node.

**Connected Graph**: A graph where a path exists between any two nodes.

**Component**: A maximal connected subgraph.

**Diameter**: The longest shortest path between any two nodes in the graph.

**Neighborhood**: The set of nodes directly connected to a given node.

**k-Hop Neighborhood**: All nodes reachable within k edges from a given node.

**Subgraph**: A graph formed from a subset of vertices and edges of another graph.

**Traversal**: The process of visiting nodes in a graph in a systematic way.

**Adjacency Matrix**: A matrix representation of a graph where entry (i,j) indicates if nodes i and j are connected.

**Adjacency List**: A representation storing for each node the list of its neighbors.

**Centrality**: A measure of the importance of a node in a graph.

**Betweenness Centrality**: How often a node appears on shortest paths between other nodes.

**PageRank**: A centrality measure based on the importance of incoming neighbors.

**Clustering Coefficient**: A measure of how much nodes cluster together.

**Community**: A group of nodes more densely connected to each other than to the rest of the graph.

**Modularity**: A measure of community structure quality.

**Triple**: A basic unit in RDF graphs: (subject, predicate, object).

**Ontology**: A formal specification of concepts and their relationships in a domain.

**Schema**: The structure defining node labels, relationship types, and their properties in a knowledge graph.

**Entity**: A distinct object or concept represented as a node in a knowledge graph.

**Entity Linking**: The task of connecting entity mentions in text to corresponding nodes in a knowledge graph.

**Relation Extraction**: Identifying relationships between entities in text.

**Knowledge Graph Completion**: Predicting missing edges in a knowledge graph.

---

### RAG-Specific Terms

**RAG Pipeline**: The sequence of steps: query → retrieval → context construction → generation.

**Retrieval-Augmented Generation**: Enhancing LLM outputs by first retrieving relevant information from external sources.

**Context**: The retrieved information provided to an LLM along with the user query.

**Context Window**: The amount of text (in tokens) an LLM can consider at once, limiting how much retrieved content can be included.

**Grounding**: Anchoring LLM responses in factual, retrieved information rather than pure generation.

**Hallucination Control**: Techniques to prevent LLMs from generating false information.

**Citation**: Attributing generated information to specific source documents.

**Source Attribution**: Identifying which retrieved documents contributed to which parts of the answer.

**Query Rewriting**: Transforming the user's query into a better form for retrieval.

**Query Expansion**: Adding related terms to the query to improve recall.

**Query Decomposition**: Breaking complex queries into simpler sub-queries.

**Multi-Hop Reasoning**: Answering questions that require connecting multiple pieces of information.

**Fusion**: Combining results from multiple retrieval methods or multiple queries.

**Reciprocal Rank Fusion** (RRF): A method to combine ranked lists from different retrieval systems.

**Metadata Filtering**: Restricting retrieval to documents matching certain attributes (date, author, type, etc.).

**Hybrid Search**: Combining different search methods (e.g., keyword + semantic).

---

### Graph RAG Terms

**GraphRAG**: RAG systems that incorporate knowledge graph reasoning alongside vector retrieval.

**Entity-Centric Retrieval**: Retrieving information focused on specific entities extracted from the query.

**Relationship-Aware Retrieval**: Using graph relationships to guide retrieval.

**Graph-Guided Retrieval**: Using knowledge graph structure to inform which documents to retrieve.

**Context Fusion**: Combining structured knowledge (from KG) with unstructured text (from RAG).

**Text-to-Cypher**: Converting natural language queries to Cypher graph queries using LLMs.

**Query Routing**: Deciding whether to use RAG, KG, or both based on query characteristics.

**Explainability Path**: A sequence of graph edges explaining how a conclusion was reached.

**Provenance Tracking**: Recording the sources (documents, graph nodes) of information in the answer.

**Confidence Score**: A measure of how certain the system is about an answer.

---

### Technical Infrastructure

**Vector Database**: A database optimized for storing and searching high-dimensional vectors.

**Index**: A data structure enabling fast search operations.

**HNSW** (Hierarchical Navigable Small World): An efficient algorithm for approximate nearest neighbor search.

**IVF** (Inverted File Index): An indexing method that partitions the vector space for faster search.

**Quantization**: Reducing vector precision to save memory (e.g., float32 → int8).

**Sharding**: Distributing data across multiple servers for scalability.

**Caching**: Storing frequently accessed results to reduce computation.

**Batch Processing**: Processing multiple items together for efficiency.

**API** (Application Programming Interface): A way for different software systems to communicate.

**Endpoint**: A specific URL where an API can be accessed.

**Latency**: The time delay between request and response.

**Throughput**: The number of requests processed per unit time.

**Rate Limiting**: Restricting the number of API requests per time period.

---

## Mathematical Notation Guide

(If mathematical notation makes you nervous, you're not alone. The good news: you don't need to be a mathematician to build RAG systems. The bad news: you can't escape notation entirely. The symbols below will appear in papers and documentation, so at minimum, you need to recognize what they mean when you see them.)

### Set Theory Notation

**∈** (Element of): x ∈ S means "x is an element of set S"
- Example: "cat" ∈ Vocabulary

**⊂** (Subset): A ⊂ B means "A is a subset of B" (all elements of A are in B)
- Example: Retrieved Documents ⊂ All Documents

**∪** (Union): A ∪ B contains all elements in A or B or both
- Example: BM25_results ∪ Dense_results

**∩** (Intersection): A ∩ B contains only elements in both A and B
- Example: Relevant ∩ Retrieved = True Positives

**∅** (Empty set): A set with no elements

**|S|** (Cardinality): The number of elements in set S
- Example: |Vocabulary| = 50,000 (vocabulary has 50,000 words)

**{x | condition}** (Set builder notation): Set of all x satisfying condition
- Example: {doc | score(doc) > 0.8} = all documents with score above 0.8

### Linear Algebra Notation

**ℝ** (Real numbers): The set of all real numbers

**ℝ^d** (d-dimensional real space): Space of vectors with d real-valued components
- Example: Embedding ∈ ℝ^768 means embedding is a 768-dimensional vector

**v** or **v** (Vector): Typically lowercase bold or with arrow
- Components: v = [v₁, v₂, ..., vₙ]

**M** or **M** (Matrix): Typically uppercase bold
- Element at row i, column j: Mᵢⱼ or M[i,j]

**v^T** (Transpose): Converts row vector to column vector or vice versa
- If v = [1, 2, 3], then v^T = [[1], [2], [3]]

**A·B** or **A**^T**B** (Dot product): Sum of element-wise products
- [1,2,3]·[4,5,6] = 1×4 + 2×5 + 3×6 = 32

**||v||** (Norm/Magnitude): Length of vector v
- ||v||₂ = √(v₁² + v₂² + ... + vₙ²) (L2 norm, Euclidean)
- ||v||₁ = |v₁| + |v₂| + ... + |vₙ| (L1 norm, Manhattan)

**⊙** (Hadamard product): Element-wise multiplication
- [1,2,3] ⊙ [4,5,6] = [4,10,18]

### Probability & Statistics Notation

**P(A)** (Probability): Probability of event A occurring
- Range: 0 ≤ P(A) ≤ 1

**P(A|B)** (Conditional probability): Probability of A given B has occurred
- Formula: P(A|B) = P(A,B) / P(B)

**P(A,B)** or **P(A∩B)** (Joint probability): Probability of both A and B occurring

**E[X]** (Expected value): Average value of random variable X
- E[X] = Σ x·P(X=x) for discrete X

**𝔼** (Expectation operator): Same as E, used in some contexts

**σ** (Standard deviation): Measure of spread in a distribution

**μ** (Mean): Average value

**Σ** (Summation): Sum over a range
- Σᵢ₌₁ⁿ xᵢ = x₁ + x₂ + ... + xₙ

**Π** (Product): Multiply over a range
- Πᵢ₌₁ⁿ xᵢ = x₁ × x₂ × ... × xₙ

**argmax** (Argument of maximum): The input that produces maximum output
- argmaxₓ f(x) = the value of x that maximizes f(x)
- Example: argmax score(doc) = document with highest score

**argmin** (Argument of minimum): The input that produces minimum output

### Calculus Notation

**∂** (Partial derivative): Derivative with respect to one variable
- ∂f/∂x = rate of change of f with respect to x

**∇** (Gradient): Vector of partial derivatives
- ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]

**∫** (Integral): Area under curve or accumulation

**≈** (Approximately equal): Two values are close but not exactly equal

**≡** (Identically equal): Two expressions are always equal

**→** (Tends to/Maps to):
- Limit: x → 0 means "x approaches 0"
- Function: f: X → Y means "f maps from X to Y"

**∞** (Infinity): Unbounded quantity

### Logic & Boolean Notation

**∧** (AND): Both conditions must be true
- A ∧ B is true only if both A and B are true

**∨** (OR): At least one condition must be true
- A ∨ B is true if A is true, or B is true, or both

**¬** (NOT): Negation
- ¬A is true if A is false

**⇒** (Implies): If A then B
- A ⇒ B means "if A is true, then B must be true"

**⇔** (If and only if): Bidirectional implication
- A ⇔ B means A ⇒ B and B ⇒ A

**∀** (For all): Universal quantifier
- ∀x ∈ S, P(x) means "for every x in S, property P(x) holds"

**∃** (There exists): Existential quantifier
- ∃x ∈ S, P(x) means "there is at least one x in S where P(x) holds"

### Graph Theory Notation

**G = (V, E)**: Graph G with vertex set V and edge set E

**V** or **V(G)**: Set of vertices/nodes in graph G

**E** or **E(G)**: Set of edges in graph G

**e = (u,v)**: Edge connecting vertices u and v

**u → v**: Directed edge from u to v

**u - v**: Undirected edge between u and v

**deg(v)**: Degree of vertex v (number of edges connected to it)

**deg⁺(v)**: Out-degree (outgoing edges in directed graph)

**deg⁻(v)**: In-degree (incoming edges in directed graph)

**N(v)**: Neighborhood of v (set of vertices adjacent to v)

**d(u,v)**: Distance between vertices u and v (length of shortest path)

**|V|**: Number of vertices (cardinality of vertex set)

**|E|**: Number of edges

**path**: Sequence of vertices [v₁, v₂, ..., vₖ] where consecutive pairs are connected

### Complexity Notation (Big-O)

**O(n)** (Big-O): Upper bound on growth rate
- O(n) = "at most proportional to n"
- O(1) = constant time
- O(log n) = logarithmic time
- O(n) = linear time
- O(n log n) = linearithmic time
- O(n²) = quadratic time
- O(2ⁿ) = exponential time

**Ω(n)** (Big-Omega): Lower bound on growth rate

**Θ(n)** (Big-Theta): Tight bound (both upper and lower)

### Common Symbols in RAG/KG Context

**q** or **q**: Query vector/text

**d** or **d**: Document vector/text

**k**: Number of results to retrieve (top-k)

**n**: Number of documents or tokens or nodes

**d** (when not document): Dimensionality of embedding space

**θ** (theta): Angle between vectors, or model parameters

**α, β, γ** (alpha, beta, gamma): Weighting coefficients
- Example: score = α·BM25 + β·Dense + γ·Recency

**λ** (lambda): Regularization parameter or weighting factor

**ϵ** (epsilon): Small positive number, error tolerance

**δ** (delta): Small change or difference

**τ** (tau): Threshold value

---

## Common Confusion: Terms That Sound Similar

**Embedding vs Encoding**:
- **Embedding**: The vector representation itself ([0.2, -0.5, ...])
- **Encoding**: The process of creating an embedding (running text through a model)

**Index vs Indexing**:
- **Index** (noun): Data structure for fast search (e.g., FAISS index)
- **Indexing** (verb): Process of adding documents to an index

**Retrieval vs Retriever**:
- **Retrieval**: The task/process of finding relevant documents
- **Retriever**: The system/component that performs retrieval

**Model vs Algorithm**:
- **Model**: Learned parameters (neural network weights)
- **Algorithm**: Step-by-step procedure (BFS, Dijkstra)

**Dense vs Sparse** (two different meanings):
- **In vectors**: Dense = most elements non-zero, Sparse = most elements zero
- **In retrieval**: Dense = learned embeddings (DPR), Sparse = keyword-based (BM25)

**Graph vs Network**:
- Generally interchangeable, but:
- **Graph**: Mathematical abstraction, formal structure
- **Network**: Often implies real-world system (social network, neural network)

**Node vs Vertex**:
- Completely interchangeable terms for the same concept
- **Node**: More common in CS/databases
- **Vertex**: More common in mathematics

**Edge vs Link vs Relationship**:
- All refer to connections between nodes
- **Edge**: Mathematical/graph theory term
- **Link**: Web/networking term
- **Relationship**: Knowledge graph/database term

**Chunk vs Passage vs Segment**:
- All refer to pieces of documents
- **Chunk**: General term, can be any size
- **Passage**: Usually paragraph-sized, coherent semantic unit
- **Segment**: Generic division of text

**Context vs Context Window**:
- **Context**: The information provided to LLM (retrieved documents + query)
- **Context Window**: The maximum token limit the LLM can process

**Latency vs Throughput**:
- **Latency**: How long one request takes (measured in milliseconds)
- **Throughput**: How many requests per second (measured in requests/sec or QPS)

**Precision vs Accuracy**:
- **Precision**: Of retrieved items, what fraction are relevant?
- **Accuracy**: Of all items, what fraction are correctly classified?
- In RAG: Precision is more important than accuracy

**Training vs Inference**:
- **Training**: Learning model parameters from data (done once, expensive)
- **Inference**: Using trained model to make predictions (done many times, needs to be fast)

**Embedding Model vs LLM**:
- **Embedding Model**: Converts text to vectors (BERT, text-embedding-3)
- **LLM**: Generates text (GPT-4, Claude)
- Some models can do both (e.g., BERT can be used for embeddings or classification)

(If you're confused about some of these distinctions, that's normal. Many of these terms won't click until you've used them in practice. The confusion is a feature, not a bug - it means you're paying attention to nuance.)

---
