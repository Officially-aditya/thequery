# 3. RAG ENGINEERING MODULE

(Welcome to the practical section. Everything before this was foundation. Everything here is production-grade engineering. The examples look simple - they're not. Each design decision has failure modes that won't appear until you hit production traffic. We'll point them out as we go.)

## What is RAG (Retrieval-Augmented Generation)?

**Problem RAG Solves**:
- LLMs have knowledge cutoff dates (trained on old data)
- LLMs hallucinate (make up facts confidently)
- LLMs can't access private/proprietary data
- LLMs have token limits (can't process entire databases)

**Solution**: Retrieve relevant information → Feed to LLM → Generate grounded answers

### RAG Pipeline (Basic)

```
User Query
    ↓
[1] Query Processing (rewrite, expand)
    ↓
[2] Retrieval (search documents)
    ↓
[3] Context Construction (format retrieved docs)
    ↓
[4] LLM Generation (answer with context)
    ↓
Answer
```

### Concrete Example

**Without RAG**:
```
User: "What was our Q4 2024 revenue?"
LLM: "I don't have access to real-time data..."
```

**With RAG**:
```
User: "What was our Q4 2024 revenue?"
    ↓
Retrieval: Find "Q4_2024_earnings.pdf"
    ↓
Context: "Q4 2024 revenue: $5.2M, up 23% YoY..."
    ↓
LLM: "According to the Q4 2024 earnings report, revenue was $5.2M, representing a 23% increase year-over-year."
```

## Chunking Strategies

### Why Chunking Matters

**Problem**: Documents are too long for:
- Embedding models (token limits: 512-8192)
- LLM context windows (need concise relevant chunks, not entire PDFs)
- Retrieval accuracy (large chunks = mixed topics = poor similarity scores)

(Chunking is the most underestimated part of RAG. Everyone obsesses over embeddings and rerankers, but bad chunking will tank your system no matter how sophisticated everything else is. A chunk that cuts off mid-sentence? The LLM gets confused. A chunk that spans three different topics? Your similarity scores are garbage. Get this right first, optimize everything else later.)

### Chunking Methods

#### 3.2.1 Fixed-Size Chunking

**Method**: Split every N characters/tokens with overlap.

```python
def fixed_size_chunking(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # Overlap prevents cutting sentences
    return chunks

document = "Long document text..." * 1000
chunks = fixed_size_chunking(document, chunk_size=500, overlap=50)
```

**Pros**: Simple, predictable size
**Cons**: Breaks mid-sentence, ignores document structure

#### 3.2.2 Sentence-Based Chunking

```python
import nltk
nltk.download('punkt')

def sentence_chunking(text, sentences_per_chunk=5):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i+sentences_per_chunk])
        chunks.append(chunk)
    return chunks
```

**Pros**: Preserves sentence boundaries
**Cons**: Variable chunk sizes

#### 3.2.3 Semantic Chunking (Advanced)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Splits on paragraph, then sentence, then word boundaries
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_text(document)
```

**Pros**: Respects document structure
**Cons**: More complex

#### 3.2.4 Structural Chunking (Best for Many Use Cases)

**Method**: Split by document structure (headers, sections, paragraphs).

```python
def structural_chunking(markdown_text):
    chunks = []
    current_chunk = ""
    current_header = ""

    for line in markdown_text.split('\n'):
        if line.startswith('#'):  # Header
            if current_chunk:
                chunks.append({
                    'content': current_chunk,
                    'header': current_header
                })
            current_header = line
            current_chunk = line + '\n'
        else:
            current_chunk += line + '\n'

    if current_chunk:
        chunks.append({'content': current_chunk, 'header': current_header})

    return chunks
```

**Pros**: Maintains semantic coherence
**Cons**: Requires structured documents

(At this point you're probably thinking "which chunking method should I use?" Here's the truth: for 80% of use cases, fixed-size with overlap (500-1000 tokens, 10-20% overlap) works fine. Don't overcomplicate it. Try the simple thing first. The advanced strategies below are for when the simple thing fails - and you'll know it has failed because your retrieval quality will be obviously bad.)

#### 3.2.5 Advanced Chunking Strategies

**Semantic Similarity-Based Chunking**:

Instead of fixed boundaries, split based on semantic coherence:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

def semantic_chunking(text, similarity_threshold=0.7):
    """
    Split text when semantic similarity between consecutive sentences drops
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = nltk.sent_tokenize(text)

    # Embed all sentences
    embeddings = model.encode(sentences)

    # Compute consecutive similarities
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = cosine_similarity(
            embeddings[i-1].reshape(1, -1),
            embeddings[i].reshape(1, -1)
        )[0][0]

        if similarity < similarity_threshold:
            # Topic changed, start new chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    # Add final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
```

**Why This Works**:
- Automatically detects topic boundaries
- No manual tuning of chunk size
- Preserves semantic coherence

**Trade-offs**:
- Computationally expensive (need to embed every sentence)
- Variable chunk sizes can be problematic for some systems
- Best for: Long documents with clear topic transitions (e.g., research papers, reports)

**Sliding Window Chunking with Context**:

```python
def sliding_window_with_context(text, window_size=500, stride=400, context_size=100):
    """
    Create overlapping chunks where each chunk includes context from previous/next
    """
    chunks = []
    metadata = []

    start = 0
    while start < len(text):
        # Main content
        end = min(start + window_size, len(text))
        chunk_text = text[start:end]

        # Add context from before
        context_before = text[max(0, start - context_size):start]

        # Add context after
        context_after = text[end:min(end + context_size, len(text))]

        # Store main chunk with metadata about context
        chunks.append({
            'main_content': chunk_text,
            'context_before': context_before,
            'context_after': context_after,
            'full_chunk': context_before + chunk_text + context_after,
            'position': (start, end)
        })

        start += stride

    return chunks

# Usage for RAG
chunks = sliding_window_with_context(document)
# Embed 'full_chunk' for better context understanding
# But retrieve 'main_content' to avoid duplication
```

**Benefits**:
- Prevents information loss at boundaries
- Each chunk has context for better embedding quality
- Answers questions that span chunk boundaries

**Hierarchical Chunking (Multi-Level)**:

```python
def hierarchical_chunking(text):
    """
    Create chunks at multiple granularities: document → section → paragraph → sentence
    """
    # Level 1: Full document summary
    doc_summary = {
        'level': 'document',
        'content': text[:1000],  # First 1000 chars as summary
        'metadata': {'type': 'overview'}
    }

    # Level 2: Sections (by headers)
    sections = text.split('\n\n')  # Simplified
    section_chunks = []

    for i, section in enumerate(sections):
        if len(section) > 100:  # Skip tiny sections
            section_chunks.append({
                'level': 'section',
                'content': section,
                'metadata': {
                    'section_id': i,
                    'parent': 'document'
                }
            })

            # Level 3: Paragraphs within section
            paragraphs = section.split('\n')
            for j, para in enumerate(paragraphs):
                if len(para) > 50:
                    section_chunks.append({
                        'level': 'paragraph',
                        'content': para,
                        'metadata': {
                            'paragraph_id': j,
                            'parent_section': i,
                            'parent': 'section'
                        }
                    })

    return [doc_summary] + section_chunks
```

**Retrieval Strategy**:
```python
# Query routing based on question type
if is_broad_question(query):
    # Retrieve from section-level chunks
    results = retrieve(query, level='section')
elif is_specific_question(query):
    # Retrieve from paragraph-level chunks
    results = retrieve(query, level='paragraph')
else:
    # Hybrid: retrieve from multiple levels
    results = retrieve(query, level='all')
```

**When to Use**:
- Complex documents with clear hierarchical structure (research papers, legal docs, manuals)
- Need to answer both broad and specific questions
- Want to provide varying levels of detail

**Entity-Aware Chunking**:

```python
import spacy

def entity_aware_chunking(text, max_chunk_size=500):
    """
    Never split named entities across chunks
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    chunks = []
    current_chunk = []
    current_size = 0

    for sent in doc.sents:
        sent_text = sent.text
        sent_size = len(sent_text)

        # Check if adding this sentence would exceed limit
        if current_size + sent_size > max_chunk_size and current_chunk:
            # Check if we're in the middle of an entity
            last_token = list(doc.sents)[len(current_chunk)-1][-1]

            if last_token.ent_type_:  # In middle of entity
                # Continue chunk to include complete entity
                current_chunk.append(sent_text)
                current_size += sent_size
            else:
                # Safe to split
                chunks.append(" ".join(current_chunk))
                current_chunk = [sent_text]
                current_size = sent_size
        else:
            current_chunk.append(sent_text)
            current_size += sent_size

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
```

**Why This Matters**:
- "Apple Inc. released..." vs "...Apple Inc. / released..." (split destroys meaning)
- Essential for queries about specific entities
- Improves retrieval accuracy for entity-centric questions

**Code-Aware Chunking** (for technical documentation):

```python
def code_aware_chunking(markdown_text):
    """
    Keep code blocks intact, never split them
    """
    chunks = []
    current_chunk = []
    in_code_block = False
    code_block = []

    for line in markdown_text.split('\n'):
        if line.strip().startswith('```'):
            if not in_code_block:
                # Starting code block
                in_code_block = True
                code_block = [line]
            else:
                # Ending code block
                code_block.append(line)
                # Add complete code block to current chunk
                current_chunk.extend(code_block)
                in_code_block = False
                code_block = []
        elif in_code_block:
            code_block.append(line)
        else:
            # Regular text
            current_chunk.append(line)

            # Check if chunk is getting too large (and not in code block)
            if len('\n'.join(current_chunk)) > 1000 and not in_code_block:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []

    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks
```

**Table-Aware Chunking**:

```python
def table_aware_chunking(text):
    """
    Extract tables separately, keep them intact
    """
    import re

    # Simple pattern for markdown tables
    table_pattern = r'\|.*\|[\s\S]*?\n\n'

    tables = []
    text_chunks = []

    # Find all tables
    for match in re.finditer(table_pattern, text):
        tables.append({
            'type': 'table',
            'content': match.group(),
            'position': match.span()
        })

    # Extract non-table text
    last_end = 0
    for table in tables:
        start, end = table['position']
        if start > last_end:
            text_chunks.append(text[last_end:start])
        last_end = end

    # Add remaining text
    if last_end < len(text):
        text_chunks.append(text[last_end:])

    # Chunk text normally, keep tables separate
    chunked_text = []
    for chunk in text_chunks:
        chunked_text.extend(fixed_size_chunking(chunk, 500, 50))

    # Combine with tables
    all_chunks = chunked_text + tables

    return all_chunks
```

### Chunking Best Practices

| Document Type | Recommended Strategy | Chunk Size | Special Considerations |
|---------------|---------------------|------------|----------------------|
| Technical docs | Structural (by headers) | 500-1000 tokens | Preserve code blocks, tables |
| Legal contracts | Sentence-based + Entity-aware | 300-500 tokens | Never split legal entities, clauses |
| News articles | Paragraph-based or Semantic | 200-400 tokens | Preserve quotes, citations |
| Code documentation | Function/class based | Varies | Keep function signatures with bodies |
| Chat logs | Fixed-size with overlap | 100-200 tokens | Maintain conversation context |
| Research papers | Hierarchical | Section: 800-1200, Para: 200-400 | Preserve sections, citations, figures |
| E-commerce products | Per-product entity chunks | 100-300 tokens | One product per chunk |
| Financial reports | Table-aware + Structural | 400-800 tokens | Keep tables intact, preserve numbers |

### The Science of Chunk Size Selection

**Too Small** (< 100 tokens):
- ❌ Lacks sufficient context
- ❌ Poor embedding quality (not enough signal)
- ❌ High retrieval cost (need more chunks to cover topic)
- ❌ Fragments coherent ideas

**Too Large** (> 1500 tokens):
- ❌ Mixed topics → poor similarity scores
- ❌ Exceeds embedding model limits
- ❌ LLM gets irrelevant information
- ❌ Slower processing

**Sweet Spot** (200-800 tokens):
- ✅ Complete semantic units
- ✅ Good embedding quality
- ✅ Manageable for LLM context
- ✅ Efficient retrieval

(Yes, there's actual math here. No, you don't need to optimize to the last token. Start with 500 tokens and 10% overlap. If your retrieval sucks, adjust. If it works, stop optimizing and ship your product.)

**Mathematical Analysis**:

```
Optimal chunk size C* minimizes:
L(C) = α·Fragmentation(C) + β·Noise(C) + γ·Cost(C)

where:
- Fragmentation(C) = E[incomplete_concepts | chunk_size=C]
- Noise(C) = E[irrelevant_content | chunk_size=C]
- Cost(C) = computational_cost(C)

Empirically:
C* ≈ 500 tokens for most domains
```

**Overlap Considerations**:

```python
# Rule of thumb: overlap = 10-20% of chunk size
chunk_size = 500
overlap = 50-100  # 10-20%

# Why overlap matters:
# Without overlap:
# Chunk 1: "...and the conclusion is"
# Chunk 2: "important for understanding..."
# ❌ Lost connection!

# With overlap:
# Chunk 1: "...and the conclusion is important for understanding..."
# Chunk 2: "...conclusion is important for understanding the next section..."
# ✅ Preserved context!
```

**Dynamic Overlap Based on Content**:

```python
def adaptive_overlap(text, base_chunk_size=500):
    """
    Increase overlap near important transitions
    """
    sentences = nltk.sent_tokenize(text)
    chunks = []

    i = 0
    while i < len(sentences):
        chunk_sents = []
        size = 0

        # Build chunk
        while size < base_chunk_size and i < len(sentences):
            chunk_sents.append(sentences[i])
            size += len(sentences[i])
            i += 1

        # Check if next sentence starts with transition word
        if i < len(sentences):
            next_sent = sentences[i].lower()
            transitions = ['however', 'moreover', 'therefore', 'in conclusion']

            if any(next_sent.startswith(t) for t in transitions):
                # Important transition - include in current chunk too
                chunk_sents.append(sentences[i])
                # Don't increment i - will be in next chunk too

        chunks.append(" ".join(chunk_sents))

    return chunks
```

### Before/After Example

**Before (Poor Chunking)**:
```
Chunk 1: "...the mitochondria is the powerhouse of the cell. It produces ATP through"
Chunk 2: "cellular respiration. In other news, photosynthesis occurs in chloroplasts..."
```
❌ Split mid-sentence, mixed topics

**After (Good Chunking)**:
```
Chunk 1: "The mitochondria is the powerhouse of the cell. It produces ATP through cellular respiration."
Chunk 2: "Photosynthesis occurs in chloroplasts and converts light energy into chemical energy."
```
✅ Complete thoughts, clear topic boundaries

## Embeddings Selection

(The embedding model you choose matters less than you think. A decent embedding model with good chunking beats a perfect embedding model with bad chunking every time. Don't spend weeks benchmarking models. Pick a reasonable one and focus on your data quality.)

### Embedding Model Comparison

| Model | Dimensions | Max Tokens | Speed | Use Case |
|-------|-----------|------------|-------|----------|
| text-embedding-3-small | 1536 | 8191 | Fast | General purpose, cost-effective |
| text-embedding-3-large | 3072 | 8191 | Medium | High accuracy needed |
| text-embedding-ada-002 | 1536 | 8191 | Fast | Legacy, still good |
| BGE-large | 1024 | 512 | Fast | Open-source, self-hosted |
| E5-mistral | 4096 | 512 | Slow | Highest quality |

### Choosing the Right Model

```python
from openai import OpenAI

client = OpenAI()

# For most cases: text-embedding-3-small
def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# For domain-specific (legal, medical): Fine-tune or use specialized models
# Example with sentence-transformers (open-source)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-large-en-v1.5')
embedding = model.encode("Your text here")
```

### Embedding Best Practices

1. **Consistency**: Use same model for indexing and querying
2. **Normalization**: Normalize embeddings for cosine similarity
3. **Metadata**: Store model version with embeddings
4. **Batch Processing**: Embed in batches for efficiency

```python
# Efficient batch embedding
def batch_embed(texts, batch_size=100):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        embeddings.extend([item.embedding for item in response.data])
    return embeddings
```

## Indexing & Vector Stores

### FAISS (Facebook AI Similarity Search)

**Use Case**: Local development, millions of vectors, no server needed

```python
import faiss
import numpy as np

# Create index
dimension = 1536  # text-embedding-3-small dimension
index = faiss.IndexFlatL2(dimension)  # L2 distance

# Add embeddings
embeddings_array = np.array(embeddings).astype('float32')
index.add(embeddings_array)

# Search
query_embedding = np.array([get_embedding(query)]).astype('float32')
k = 5  # Top 5 results
distances, indices = index.search(query_embedding, k)

# indices[0] contains IDs of top 5 similar chunks
```

**Advanced FAISS** (for scale):
```python
# IVF index for faster search (100M+ vectors)
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, 100)  # 100 clusters

# Train index (required for IVF)
index.train(embeddings_array)
index.add(embeddings_array)
```

### ChromaDB

**Use Case**: Simple prototyping, built-in embedding, metadata filtering

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.create_collection(
    name="my_documents",
    metadata={"description": "Company knowledge base"}
)

# Add documents (auto-embedding)
collection.add(
    documents=["RAG is powerful", "Knowledge graphs structure data"],
    metadatas=[{"source": "blog", "date": "2024-01"},
               {"source": "paper", "date": "2024-02"}],
    ids=["doc1", "doc2"]
)

# Query with metadata filter
results = collection.query(
    query_texts=["what is RAG?"],
    n_results=5,
    where={"source": "blog"}  # Metadata filter
)
```

### Pinecone (Production)

**Use Case**: Production deployment, auto-scaling, managed service

```python
import pinecone

# Initialize
pinecone.init(api_key="your-api-key", environment="us-west1-gcp")
index = pinecone.Index("knowledge-base")

# Upsert vectors
index.upsert(vectors=[
    ("doc1", embedding1, {"text": "...", "source": "..."}),
    ("doc2", embedding2, {"text": "...", "source": "..."})
])

# Query
results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True,
    filter={"source": {"$eq": "blog"}}
)
```

## Retrievers (BM25, Hybrid, Dense)

### BM25 Retriever

```python
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents
        tokenized = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query, top_k=5):
        scores = self.bm25.get_scores(query.split())
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]
```

### Dense Retriever (Semantic)

```python
class DenseRetriever:
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.index = faiss.IndexFlatL2(len(embeddings[0]))
        self.index.add(np.array(embeddings).astype('float32'))

    def retrieve(self, query, top_k=5):
        query_emb = get_embedding(query)
        distances, indices = self.index.search(
            np.array([query_emb]).astype('float32'), top_k
        )
        return [self.documents[i] for i in indices[0]]
```

### Hybrid Retriever (Best of Both)

```python
class HybridRetriever:
    def __init__(self, documents, embeddings):
        self.bm25 = BM25Retriever(documents)
        self.dense = DenseRetriever(documents, embeddings)
        self.documents = documents

    def retrieve(self, query, top_k=5, alpha=0.7):
        # Get candidates from both
        bm25_docs = self.bm25.retrieve(query, top_k=20)
        dense_docs = self.dense.retrieve(query, top_k=20)

        # Score combination (simplified)
        scores = {}
        for doc in bm25_docs:
            scores[doc] = scores.get(doc, 0) + (1 - alpha)
        for doc in dense_docs:
            scores[doc] = scores.get(doc, 0) + alpha

        # Return top-k
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in sorted_docs[:top_k]]
```

**Why Hybrid?**
- BM25 catches exact term matches ("Product ID: ABC123")
- Dense catches semantic similarity ("car" → "automobile")
- Together: Best recall

## Rerankers (Cross-Encoders)

### Why Reranking?

**Problem**: Initial retrieval optimizes for speed (approximate search). Reranking adds precision.

(Reranking feels like overkill when you're prototyping. It's not. Your initial retrieval will return garbage in the top 5 about 30% of the time. Reranking fixes that. Skip it if you want, but don't be surprised when your users complain that the chatbot gives them irrelevant answers.)

**Pipeline**:
```
Query → Retrieve 100 candidates → Rerank to top 5 → Pass to LLM
```

### Cross-Encoder Reranking

```python
from sentence_transformers import CrossEncoder

# Load reranker model
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query, documents, top_k=5):
    # Create query-document pairs
    pairs = [[query, doc] for doc in documents]

    # Score all pairs
    scores = reranker.predict(pairs)

    # Sort and return top-k
    sorted_indices = np.argsort(scores)[-top_k:][::-1]
    return [documents[i] for i in sorted_indices]

# Usage
initial_results = retriever.retrieve(query, top_k=100)
final_results = rerank(query, initial_results, top_k=5)
```

### Before/After Reranking

**Before (Just Dense Retrieval)**:
```
Query: "How do I reset my password?"
Results:
1. "Password security best practices..." (semantic match, but wrong)
2. "Creating strong passwords..." (semantic match, but wrong)
3. "To reset your password, click..." (correct, but ranked 3rd)
```

**After (With Reranker)**:
```
Results:
1. "To reset your password, click..." ✅
2. "Password reset troubleshooting..." ✅
3. "Password security best practices..."
```

## Query Rewriting & Decomposition

### Query Rewriting

**Goal**: Transform user query into better retrieval queries.

```python
def rewrite_query(user_query):
    prompt = f"""
    Rewrite this user query to be more effective for document retrieval.
    Make it clearer and add important keywords.

    Original query: {user_query}

    Rewritten query:
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content

# Example
user_query = "How do I fix it?"
rewritten = rewrite_query(user_query)
# Result: "How to troubleshoot and fix common system errors"
```

### Query Decomposition (Multi-Step Queries)

**Use Case**: Complex questions requiring multiple retrievals

```python
def decompose_query(complex_query):
    prompt = f"""
    Break this complex question into simpler sub-questions:

    Question: {complex_query}

    Sub-questions (as JSON list):
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return eval(response.choices[0].message.content)

# Example
query = "Compare the performance of GPT-4 and Claude on coding tasks"
sub_queries = decompose_query(query)
# Result: [
#   "What is the performance of GPT-4 on coding tasks?",
#   "What is the performance of Claude on coding tasks?",
#   "How do GPT-4 and Claude compare overall?"
# ]

# Retrieve for each sub-query
all_docs = []
for sq in sub_queries:
    docs = retriever.retrieve(sq)
    all_docs.extend(docs)
```

## Context Window Optimization

### Context Construction

```python
def build_context(retrieved_docs, max_tokens=4000):
    context = ""
    token_count = 0

    for i, doc in enumerate(retrieved_docs):
        doc_tokens = len(doc.split()) * 1.3  # Rough estimate

        if token_count + doc_tokens > max_tokens:
            break

        context += f"\n[Document {i+1}]\n{doc}\n"
        token_count += doc_tokens

    return context

# Usage
prompt = f"""
Answer the question using only the context below.

Context:
{build_context(retrieved_docs)}

Question: {user_query}

Answer:
"""
```

### Sliding Window Retrieval

For very long documents:

```python
def sliding_window_retrieval(long_document, query, window_size=500, stride=250):
    chunks = []
    for i in range(0, len(long_document), stride):
        chunk = long_document[i:i+window_size]
        chunks.append(chunk)

    # Embed and retrieve as normal
    chunk_embeddings = [get_embedding(c) for c in chunks]
    # ... retrieve top chunks
```

## Cited Answers & Hallucination Control

(This is the difference between a demo and a product. Demos can hallucinate and nobody cares. Products that hallucinate get you sued, fired, or worse. Force citations. Always. If the LLM can't cite a source, it shouldn't make the claim. This is not negotiable for production systems.)

### Citation Pattern

```python
def rag_with_citations(query, retrieved_docs):
    # Build context with source IDs
    context = ""
    for i, doc in enumerate(retrieved_docs):
        context += f"\n[Source {i+1}]: {doc['text']}\n"

    prompt = f"""
    Answer the question using the provided sources.
    Cite sources using [Source X] notation.

    Context:
    {context}

    Question: {query}

    Answer (with citations):
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content

# Example output:
# "The company's Q4 revenue was $5.2M [Source 1], representing
#  a 23% increase from Q3 [Source 2]."
```

### Hallucination Detection

```python
def detect_hallucination(answer, context):
    prompt = f"""
    Check if the answer is fully supported by the context.

    Context: {context}
    Answer: {answer}

    Is the answer grounded in the context? (Yes/No)
    If No, list the unsupported claims.

    Response:
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
```

### "I Don't Know" Pattern

```python
prompt = f"""
Answer the question using only the context below.

IMPORTANT:
- If the answer is not in the context, respond with "I don't have enough information to answer this question."
- Do not make up or infer information not explicitly stated.

Context:
{context}

Question: {query}

Answer:
"""
```

(You now know how to build a production RAG system. Not a toy, not a demo - a real system that handles actual user queries without hallucinating nonsense. The next section covers knowledge graphs, which will let you add structured reasoning on top of your unstructured retrieval. This is where things get interesting.)

---
